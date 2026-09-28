"""
Applies an integrator baked by `bake_for_batch.py` to detector frames.

Frames are `(x, y, frame...)`, or `(x, y, module, frame...)` for a multi-module
detector baked as `(x, module * y)`. The output is `(npt0, frame...)` for 1D
and `(npt0, npt1, frame...)` for 2D, i.e. radial is the first dimension.
"""

const Radial    = Dim{:radial}
const Azimuthal = Dim{:azimuthal}
const Frame     = Dim{:frame}

"""
    BakedIntegrator

An integrator baked from a configured `pyFAI.AzimuthalIntegrator`. Holds the
frozen CSR sparse matrix and correction weights for either 1D or 2D azimuthal
integration, along with the bin centres and unit strings for the output axes.
Construct one with [`load_baked`](@ref), or directly from an integrator with
`BakedIntegrator(ai, npt; kwargs...)` when PythonCall is loaded, and apply it
with [`integrate`](@ref).
"""
@kwdef struct BakedIntegrator
    # pyFAI's CSR arrays, shifted to 1-based
    colptr::Vector{Int32}
    rowval::Vector{Int32}
    raw_nz::Vector{Float32}
    corr_nz::Vector{Float32}

    bin_centers0::Vector{Float32}            # radial axis
    bin_centers1::Vector{Float32}            # azimuthal axis (empty for 1D)
    shape::Tuple{Vararg{Int}}                # (W, H)
    unit0::String
    unit1::String                            # "" for 1D
    split::String
    npt0::Int                                # radial bins
    npt1::Int                                # azimuthal bins (0 for 1D)
    ndim::Int                                # 1 or 2
end

function Base.show(io::IO, b::BakedIntegrator)
    if b.ndim == 1
        print(io, BakedIntegrator, "(1D, $(b.shape), npt=$(b.npt0))")
    else
        print(io, BakedIntegrator, "(2D, $(b.shape), npt=($(b.npt0), $(b.npt1)))")
    end
end

function Base.:(==)(a::BakedIntegrator, b::BakedIntegrator)
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(BakedIntegrator))
end

function Base.hash(b::BakedIntegrator, h::UInt)
    for f in fieldnames(BakedIntegrator)
        h = hash(getfield(b, f), h)
    end

    h
end

# `get(T, key)` reads one field from an HDF5 file or Python dict.
function _baked_from(get)
    version = get(Int, "format_version")
    if version != 2
        error("unsupported baked integrator format_version $version, re-bake with the current bake_for_batch.py")
    end

    dummy = get(Float32, "dummy")
    delta_dummy = get(Float32, "delta_dummy")
    if (isfinite(dummy) && dummy != 0) || (isfinite(delta_dummy) && delta_dummy != 0)
        @warn "baked integrator has nonzero pyFAI dummy/delta_dummy; \
               these are NOT applied here, so I(q) may differ from \
               ai.integrate1d on pixels matching the dummy sentinel" dummy delta_dummy
    end

    ndim    = get(Int, "ndim")
    shape_c = get(Vector{Int}, "shape")
    length(shape_c) == 2 ||
        error("only 2D detector shapes are supported here, got $shape_c")
    H, W = shape_c

    bin_centers1, unit1, npt1 = if ndim == 2
        get(Vector{Float32}, "bin_centers1"), get(String, "unit1"), get(Int, "npt1")
    else
        Float32[], "", 0
    end

    # Reversed so that `vec(frame)` matches pyFAI's C-order pixel indices
    BakedIntegrator(;
        colptr=get(Vector{Int32}, "indptr") .+ Int32(1),
        rowval=get(Vector{Int32}, "indices") .+ Int32(1),
        raw_nz=get(Vector{Float32}, "data_raw"),
        corr_nz=get(Vector{Float32}, "data_corr"),
        bin_centers0=get(Vector{Float32}, "bin_centers0"),
        bin_centers1,
        shape=(W, H),
        unit0=get(String, "unit0"),
        unit1,
        split=get(String, "split"),
        npt0=get(Int, "npt0"),
        npt1,
        ndim,
    )
end

# Stored as HDF5 attributes; everything else is a dataset.
const _BAKED_ATTRS = ("shape", "ndim", "unit0", "unit1", "split", "npt0", "npt1",
                      "format_version", "dummy", "delta_dummy")

"""
    load_baked(path::AbstractString)

Load a [`BakedIntegrator`](@ref) from an HDF5 file written by
`bake_for_batch.write_hdf5(...)`. Warns if the bake has a nonzero
`dummy`/`delta_dummy`, since the dummy mask is not applied.
"""
function load_baked(path::AbstractString)
    h5open(path, "r") do f
        get(::Type{T}, key) where {T} =
            if key in _BAKED_ATTRS
                raw = read_attribute(f, key)
                raw isa T ? raw : T(raw)
            else
                read(f[key])::T
            end

        _baked_from(get)
    end
end

function _meta(b::BakedIntegrator)
    md = Dict("unit0" => b.unit0, "split" => b.split)
    if b.ndim == 2
        md["unit1"] = b.unit1
    end

    return md
end

# Trailing dims are copied from a DimArray input, otherwise they're named
# `Frame` if there's one or `dim_N` if there's more.
function _wrap(b::BakedIntegrator, out::AbstractArray{Float32}, extra_shape::Tuple, frames,
               nd_frame::Int)
    trail_dims = if frames isa AbstractDimArray && !isempty(extra_shape)
        otherdims(frames, ntuple(identity, nd_frame))
    elseif isempty(extra_shape)
        ()
    elseif length(extra_shape) == 1
        (Frame(1:extra_shape[1]),)
    else
        ntuple(i -> Dim{Symbol(:dim_, i + nd_frame)}(1:extra_shape[i]),
               length(extra_shape))
    end

    if b.ndim == 1
        DimArray(reshape(out, b.npt0, extra_shape...),
                 (Radial(b.bin_centers0), trail_dims...);
                 name=:intensity, metadata=_meta(b))
    else
        DimArray(reshape(out, b.npt0, b.npt1, extra_shape...),
                 (Radial(b.bin_centers0), Azimuthal(b.bin_centers1),
                  trail_dims...);
                 name=:intensity, metadata=_meta(b))
    end
end

"""
    output_size(b::BakedIntegrator, frames::AbstractArray)

Calculate the size of the output array needed for `integrate!(out, b, frames)`.
"""
output_size(b::BakedIntegrator) = b.ndim == 2 ? (b.npt0, b.npt1) : (b.npt0,)

function output_size(b::BakedIntegrator, frames::AbstractArray)
    extra_size = size(frames)[_checked_frame_ndims(b, size(frames)) + 1:end]
    (output_size(b)..., extra_size...)
end

# Length of the shortest prefix of `sz` that starts with `x` and flattens to
# `b.shape`, or `nothing`.
function _frame_ndims(b::BakedIntegrator, sz::Dims)
    npix = prod(b.shape)
    if !isempty(sz) && sz[1] == b.shape[1]
        p = 1
        for (i, s) in enumerate(sz)
            p *= s
            if p == npix
                return i
            end
        end
    end

    return nothing
end

function _checked_frame_ndims(b::BakedIntegrator, sz::Dims)
    nd = _frame_ndims(b, sz)
    if isnothing(nd)
        throw(DimensionMismatch("frame size $sz does not start with dims flattening to baked shape $(b.shape)"))
    end

    return nd
end

# `strides` throws on arrays that aren't strided
function _is_contiguous(A::AbstractArray)
    s = try
        strides(A)
    catch e
        if e isa ArgumentError || e isa MethodError
            return false
        end
        rethrow()
    end

    return s == cumprod((1, size(A)[1:end-1]...))
end

_offset(I::CartesianIndex, s::Tuple) = sum((Tuple(I) .- 1) .* s; init=0)

# Integrate a single frame, skipping non-finite pixels
function _fused_spmv!(I::AbstractVector{Float32}, b::BakedIntegrator,
                      rowval::AbstractVector{<:Integer}, x::AbstractVector, offset::Int)
    colptr = b.colptr
    raw_nz = b.raw_nz
    corr_nz = b.corr_nz

    @inbounds for bin in eachindex(I)
        s = 0.0
        n = 0.0

        for p in colptr[bin]:colptr[bin+1]-1
            v   = x[rowval[p] + offset]
            ok  = isfinite(v)

            s = muladd(raw_nz[p],  ifelse(ok, v,   0.0), s)
            n = muladd(corr_nz[p], ifelse(ok, 1.0, 0.0), n)
        end

        I[bin] = s / n
    end
end

"""
    allocate_output(b::BakedIntegrator, frame::AbstractMatrix)
    allocate_output(b::BakedIntegrator, frames::AbstractArray)

Allocate an output `Array` to be used with `integrate!()`.
"""
function allocate_output(b::BakedIntegrator, frames::AbstractArray)
    Array{Float32}(undef, output_size(b, frames))
end

function _integrate!(out::AbstractArray{Float32}, b::BakedIntegrator, frames::AbstractArray;
                     scheduler::Symbol=:static, chunksize=nothing)
    if ndims(frames) == 1
        throw(ArgumentError("A vector was passed as `frame`, but it needs to be a 2D array with shape $(b.shape)"))
    end
    nd_frame = _checked_frame_ndims(b, size(frames))
    extra_shape = size(frames)[nd_frame + 1:end]
    expected_size = (output_size(b)..., extra_shape...)
    if size(out) != expected_size
        throw(DimensionMismatch("out size $(size(out)) ≠ (nbins, frame...) = $expected_size"))
    end

    src = frames isa AbstractDimArray ? parent(frames) : frames
    mem = src isa PermutedDimsArray ? parent(src) : src
    if !_is_contiguous(mem)
        throw(ArgumentError("frames must be contiguous in memory, or a PermutedDimsArray of a contiguous array"))
    end

    # Pixel `pix` of frame `k` is `vec(mem)[rowval[pix] + offsets[k]]`
    s = strides(src)
    frame_sz, frame_s = size(src)[1:nd_frame], s[1:nd_frame]
    rowval = if frame_s == cumprod((1, frame_sz[1:end-1]...))
        b.rowval
    else
        pixel_offset = [1 + _offset(I, frame_s) for I in CartesianIndices(frame_sz)]
        pixel_offset[b.rowval]
    end
    offsets = vec([_offset(J, s[nd_frame+1:end]) for J in CartesianIndices(extra_shape)])

    Y = reshape(out, prod(output_size(b)), :)
    _integrate_frames!(Y, b, rowval, vec(mem), offsets, scheduler, chunksize)

    _wrap(b, out, extra_shape, frames, nd_frame)
end

function _integrate_frames!(Y, b, rowval, x, offsets, scheduler, chunksize)
    if length(offsets) == 1
        _fused_spmv!(@view(Y[:, 1]), b, rowval, x, offsets[1])
    else
        @tasks for k in eachindex(offsets)
            @set begin
                scheduler = scheduler
                chunksize = chunksize
            end

            _fused_spmv!(@view(Y[:, k]), b, rowval, x, offsets[k])
        end
    end
end

"""
    integrate!(out::AbstractArray{Float32}, b, frames::AbstractArray;
               scheduler=:static, chunksize=nothing)

Integrate `frames` of shape `(x, y, frame...)` into `out` of shape
`(output_size(b)..., frame...)`, one frame per task via OhMyThreads. For a
multi-module detector baked as `(x, module * y)` the frames can also be
`(x, y, module, frame...)`.

`frames` must be contiguous, or a `PermutedDimsArray` of a contiguous array,
e.g. `PermutedDimsArray(data, (1, 2, 4, 3))` for module-major data. A single
frame is integrated on the calling task. Returns a `DimArray` view wrapping
`out`.
"""
integrate!(args...; kwargs...) = _integrate!(args...; kwargs...)

function _integrate(b::BakedIntegrator, frames::AbstractArray;
                    scheduler::Symbol=:static, chunksize=nothing)
    out = allocate_output(b, frames)
    _integrate!(out, b, frames; scheduler, chunksize)
end

"""
    integrate(b, frames::AbstractArray; scheduler=:static, chunksize=nothing)

Allocate the output with [`allocate_output`](@ref) and forward to
[`integrate!`](@ref).
"""
integrate(args...; kwargs...) = _integrate(args...; kwargs...)
