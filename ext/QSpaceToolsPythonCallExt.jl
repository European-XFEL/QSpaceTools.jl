module QSpaceToolsPythonCallExt

using QSpaceTools: QSpaceTools
using LinearAlgebra: normalize
using PythonCall: Py, PyArray, PyIterable, pyconvert, pyimport, pyisinstance, pybuiltins, pytype

function QSpaceTools.load_baked(baked::Py)
    if !pyisinstance(baked, pybuiltins.dict)
        throw(ArgumentError("load_baked(::Py) expected the dict returned by bake_for_batch, got a $(pytype(baked))"))
    end

    dummy = pyconvert(Float32, baked["dummy"])
    delta_dummy = pyconvert(Float32, baked["delta_dummy"])
    if (isfinite(dummy) && dummy != 0) || (isfinite(delta_dummy) && delta_dummy != 0)
        @warn "baked integrator has nonzero pyFAI dummy/delta_dummy; \
               these are NOT applied here, so I(q) may differ from \
               ai.integrate1d on pixels matching the dummy sentinel" dummy delta_dummy
    end

    ndim = pyconvert(Int, baked["ndim"])
    shape_c = pyconvert(Vector{Int}, baked["shape"])
    if length(shape_c) != 2
        error("only 2D detector shapes are supported here, got $shape_c")
    end
    H, W = shape_c

    bin_centers1, unit1, npt1 = if ndim == 2
        (pyconvert(Vector{Float32}, baked["bin_centers1"]),
         pyconvert(String, baked["unit1"]),
         pyconvert(Int, baked["npt1"]))
    else
        Float32[], "", 0
    end

    # Reversed so that `vec(frame)` matches pyFAI's C-order pixel indices
    QSpaceTools.BakedIntegrator(;
        colptr=pyconvert(Vector{Int32}, baked["indptr"]) .+ Int32(1),
        rowval=pyconvert(Vector{Int32}, baked["indices"]) .+ Int32(1),
        raw_nz=pyconvert(Vector{Float32}, baked["data_raw"]),
        corr_nz=pyconvert(Vector{Float32}, baked["data_corr"]),
        bin_centers0=pyconvert(Vector{Float32}, baked["bin_centers0"]),
        bin_centers1,
        shape=(W, H),
        unit0=pyconvert(String, baked["unit0"]),
        unit1,
        split=pyconvert(String, baked["split"]),
        npt0=pyconvert(Int, baked["npt0"]),
        npt1,
        ndim,
    )
end

_bake_module::Union{Py, Nothing} = nothing

"""
    BakedIntegrator(ai::Py, npt; kwargs...)

Bake a pyFAI `AzimuthalIntegrator` in memory. `npt` is the number of radial
bins for 1D or `(nrad, nazim)` for 2D, and `kwargs` are passed to
`bake_for_batch()`.
"""
function QSpaceTools.BakedIntegrator(ai::Py, npt; kwargs...)
    # Checked by name so that pyFAI isn't imported for the check
    is_ai = any(pytype(ai).__mro__) do cls
        class_fqn(cls) == "pyFAI.integrator.azimuthal.AzimuthalIntegrator"
    end
    if !is_ai
        throw(ArgumentError("BakedIntegrator(::Py) expected a pyFAI AzimuthalIntegrator, got a $(python_fqn(ai))"))
    end

    # bake_for_batch.py lives at the package root, import it from there
    if isnothing(_bake_module)
        path = joinpath(pkgdir(QSpaceTools), "bake_for_batch.py")
        util = pyimport("importlib.util")
        spec = util.spec_from_file_location("qspacetools_bake_for_batch", path)
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        global _bake_module = mod
    end

    QSpaceTools.load_baked(_bake_module.bake_for_batch(ai, npt; kwargs...))
end

# Function barrier for the zero-copy numpy wrapper `pos`.
function _lab_positions(pos::AbstractArray, R::QSpaceTools.Mat3,
                        offset::QSpaceTools.Vec3, npix::Int)
    src = reshape(pos, 3, npix)
    positions = Matrix{Float64}(undef, 3, npix)
    @inbounds for k in 1:npix
        p = R * (QSpaceTools.Vec3(src[1, k], src[2, k], src[3, k]) + offset)
        positions[1, k] = p[1]
        positions[2, k] = p[2]
        positions[3, k] = p[3]
    end
    return positions
end

class_fqn(cls::Py) = join(string.([cls.__module__, cls.__qualname__]), ".")
python_fqn(obj::Py) = class_fqn(obj.__class__)
python_class(obj::Py) = string(obj.__class__.__qualname__)

"""
    Geometry(geom; distance, sample_axes, detector_axes, beam_direction,
             wavelength, extra_geom_axes=("y+", "z+", "x+"), center=nothing,
             sample_normal=nothing, sample_faceup=nothing)

Build a [`Geometry`](@ref) from an EXtra-geom detector geometry, one position
per pixel, so multi-module detectors need no assembly.

`extra_geom_axes` gives the lab-frame directions of EXtra-geom's `(x, y, z)`
axes, in the same notation as the other axis arguments. EXtra-geom's `z` is its
beam axis, so the third entry must agree with `beam_direction`.

`distance` is the sample-detector distance **in metres**: EXtra-geom's `z`
coordinates are only relative module offsets, so it is added along the beam
before mapping into the lab frame.

`data_shape` is EXtra-geom's `(nmodules, ss, fs)` reversed to `(fs, ss,
nmodules)`, matching the numpy layout pixel for pixel. **Frames passed to
`rss`/`rsm` must already be transposed** into that order; `center` and
`image_axes` are reported in the same reversed order. `center` defaults to the
middle of the snapped detector image.
"""
function QSpaceTools.Geometry(geom::Py;
                              distance,
                              sample_axes,
                              detector_axes,
                              beam_direction,
                              wavelength,
                              extra_geom_axes=("y+", "z+", "x+"),
                              center=nothing,
                              sample_normal=nothing,
                              sample_faceup=nothing)
    base = pyimport("extra_geom.base")
    if !pyisinstance(geom, base.DetectorGeometryBase)
        throw(ArgumentError("Geometry(::Py) expected an EXtra-geom DetectorGeometryBase, got a $(pytype(geom))"))
    end

    # Columns are the lab-frame images of EXtra-geom's x, y and z.
    ax = map(QSpaceTools.parse_axis, extra_geom_axes)
    R = QSpaceTools.Mat3(hcat(ax...))
    beam = normalize(QSpaceTools.parse_axis(beam_direction))
    if !isapprox(ax[3], beam; atol=1e-12)
        throw(ArgumentError("extra_geom_axes[3] $(ax[3]) is EXtra-geom's beam axis and must match beam_direction $beam"))
    end

    d = Float64(distance)

    # `.T` of the C-order (nmodules, ss, fs, 3) array is a (3, fs, ss, nmodules)
    # view in Julia layout, with no permute.
    pos = PyArray(geom.get_pixel_positions().T)
    shape = reverse(pyconvert(NTuple{3, Int}, geom.expected_data_shape))
    npix = prod(shape)
    positions = _lab_positions(pos, R, QSpaceTools.Vec3(0.0, 0.0, d), npix)

    px = pyconvert(Float64, geom.pixel_size)

    # The snapped image's rows run along physical +y and its columns along +x;
    # reversed here like everything else.
    image_axes = (R * QSpaceTools.AXIS_VECS["x+"], R * QSpaceTools.AXIS_VECS["y+"])

    c = if isnothing(center)
        reverse(pyconvert(NTuple{2, Int}, geom._snapped().size_yx)) ./ 2
    else
        (Float64(center[1]), Float64(center[2]))
    end

    return QSpaceTools.Geometry(positions, shape;
                                sample_axes, detector_axes, image_axes,
                                beam_direction, sample_normal, sample_faceup,
                                pixel_size=(px, px), center=c, distance=d,
                                wavelength)
end

function get_py_array(frames::Py)
    if python_fqn(frames) == "xarray.core.dataarray.DataArray"
        frames = frames.values.T
    elseif python_fqn(frames) == "numpy.ndarray"
        frames = frames.T
    else
        throw(ArgumentError("frames array must be convertible to a PyArray, this type is not supported: $(python_class(frames))"))
    end

    # A permuted view of a contiguous array, e.g. from `np.swapaxes`, becomes a
    # PermutedDimsArray of the contiguous array
    if !pyconvert(Bool, frames.flags.f_contiguous)
        order = sortperm(pyconvert(Vector{Int}, frames.strides))
        contiguous = frames.transpose(Tuple(order .- 1))
        if pyconvert(Bool, contiguous.flags.f_contiguous)
            return PermutedDimsArray(PyArray(contiguous), invperm(order))
        end
    end

    PyArray(frames)
end

get_py_array(frames::Union{PyIterable, PyArray}) = get_py_array(frames.py)

const py_types = Union{Py, PyIterable, PyArray}

# If frames is a PyArray of the right shape then we don't need to do anything
function get_py_array(frames::py_types, frame_shape::Tuple)
    if frames isa PyArray && size(frames)[1:length(frame_shape)] == frame_shape
        frames
    else
        get_py_array(frames)
    end
end

function QSpaceTools.rsm(frames::py_types, geom::QSpaceTools.Geometry; kwargs...)
    QSpaceTools._rsm(get_py_array(frames, geom.data_shape), geom; kwargs...)
end

function QSpaceTools.rsm!(outputs::Union{AbstractArray{Float64, 3}, QSpaceTools.QProjections},
                          frames::py_types,
                          geom::QSpaceTools.Geometry,
                          args...; kwargs...)
    QSpaceTools._rsm!(outputs, get_py_array(frames, geom.data_shape), geom, args...; kwargs...)
end

function get_py_array(frames::py_types, b::QSpaceTools.BakedIntegrator)
    if frames isa PyArray && !isnothing(QSpaceTools._frame_ndims(b, size(frames)))
        frames
    else
        get_py_array(frames)
    end
end

function QSpaceTools.integrate(b::QSpaceTools.BakedIntegrator, frames::py_types; kwargs...)
    QSpaceTools._integrate(b, get_py_array(frames, b); kwargs...)
end

function QSpaceTools.integrate!(out::AbstractArray{Float32}, b::QSpaceTools.BakedIntegrator,
                                frames::py_types; kwargs...)
    QSpaceTools._integrate!(out, b, get_py_array(frames, b); kwargs...)
end

end # module QSpaceToolsPythonCallExt
