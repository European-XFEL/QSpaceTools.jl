module QSpaceToolsPythonCallExt

using QSpaceTools: QSpaceTools
using LinearAlgebra: normalize
using PythonCall: Py, PyArray, PyIterable, pyconvert, pyimport, pyisinstance, pybuiltins, pytype

# Build a BakedIntegrator from the `baked` dict returned by `bake_for_batch`,
# skipping HDF5.
function QSpaceTools.load_baked(baked::Py)
    if !pyisinstance(baked, pybuiltins.dict)
        throw(ArgumentError("load_baked(::Py) expected the dict returned by bake_for_batch, got a $(pytype(baked))"))
    end

    QSpaceTools._baked_from((T, key) -> pyconvert(T, baked[key]))
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

python_fqn(obj::Py) = join(string.([obj.__class__.__module__, python_class(obj)]), ".")
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

    PyArray(frames)
end

get_py_array(frames::Union{PyIterable, PyArray}) = get_py_array(frames.py)

const py_types = Union{Py, PyIterable, PyArray}

QSpaceTools.rsm(frames::py_types, geom::QSpaceTools.Geometry; kwargs...) = QSpaceTools._rsm(get_py_array(frames), geom; kwargs...)

function QSpaceTools.rsm!(outputs::Union{AbstractArray{Float64, 3}, QSpaceTools.QProjections},
                          frames::py_types,
                          geom::QSpaceTools.Geometry,
                          args...; kwargs...)
    # If frames is a PyArray of the right shape then we don't need to do anything
    if !(frames isa PyArray && size(frames)[1:length(geom.data_shape)] == geom.data_shape)
        frames = get_py_array(frames)
    end

    QSpaceTools._rsm!(outputs, frames, geom, args...; kwargs...)
end

end # module QSpaceToolsPythonCallExt
