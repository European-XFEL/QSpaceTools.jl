const Qx = Dim{:qx}
const Qy = Dim{:qy}
const Qz = Dim{:qz}
const Q_DIMS = (Qx, Qy, Qz)

# Which two q-components each projection grids.
const _PROJECTION_PAIRS = ((1, 2), (1, 3), (2, 3))

# Projection spec to q-component indices.
const _Q_AXIS_IDX = (qx=1, qy=2, qz=3)
_proj_indices(p::Tuple{Symbol, Symbol}) = (_Q_AXIS_IDX[p[1]], _Q_AXIS_IDX[p[2]])
_proj_indices(p::Tuple{Integer, Integer}) = (Int(p[1]), Int(p[2]))

"""
    QProjections(qxqy, qxqz, qyqz)

The three axis-pair projections of a reciprocal space map, as returned by
`rsm(...; output=:projections)` and [`allocate_output`](@ref).
"""
struct QProjections{XY <: AbstractMatrix, XZ <: AbstractMatrix, YZ <: AbstractMatrix}
    qxqy::XY
    qxqz::XZ
    qyqz::YZ
end

_grids(p::QProjections) = (p.qxqy, p.qxqz, p.qyqz)

Base.:(==)(a::QProjections, b::QProjections) = all(map(==, _grids(a), _grids(b)))
Base.isapprox(a::QProjections, b::QProjections; kwargs...) =
    all(map((x, y) -> isapprox(x, y; kwargs...), _grids(a), _grids(b)))

# Bin count per q-component implied by output buffers.
_gridder_size(grid::AbstractArray) = size(grid)

function _gridder_size(p::QProjections)
    nx, ny = size(p.qxqy)
    nx2, nz = size(p.qxqz)
    ny2, nz2 = size(p.qyqz)
    if (nx2, ny2, nz2) != (nx, ny, nz)
        throw(ArgumentError("projection sizes disagree: qxqy $(size(p.qxqy)), " *
                            "qxqz $(size(p.qxqz)), qyqz $(size(p.qyqz))"))
    end
    return (nx, ny, nz)
end

# Buffers for one grid, filled by output range: each task owns a range of the
# last dimension and reads every point, so nothing is summed afterwards. The
# numerator lives here so that summing the map leaves it intact.
struct GridBuffers{D}
    grid::Array{Float64, D}
    gridder::GridderWorkspace{D}
end

GridBuffers(size::NTuple{D, Int}) where {D} =
    GridBuffers{D}(Array{Float64, D}(undef, size...), GridderWorkspace(size))

# Buffers for the three projection grids, filled by point chunk: the grids
# share no axis to split, so each task accumulates its chunk into its own
# copy of every grid and norm, summed by `_sum_task_buffers!`. Copies are
# added on demand and kept.
struct ProjectionBuffers
    sizes::NTuple{3, NTuple{2, Int}}
    grids::Vector{NTuple{3, Matrix{Float64}}}    # [task][grid]
    norms::Vector{NTuple{3, Matrix{Float64}}}
end

function ProjectionBuffers(n::NTuple{3, Int})
    sizes = map(pair -> (n[pair[1]], n[pair[2]]), _PROJECTION_PAIRS)
    return ProjectionBuffers(sizes, NTuple{3, Matrix{Float64}}[], NTuple{3, Matrix{Float64}}[])
end

function _ensure_task_buffers!(buffers::ProjectionBuffers, n::Integer)
    while length(buffers.grids) < n
        push!(buffers.grids, map(sz -> zeros(Float64, sz), buffers.sizes))
        push!(buffers.norms, map(sz -> zeros(Float64, sz), buffers.sizes))
    end
    return nothing
end

_allocate_output(buffers::GridBuffers) = similar(buffers.grid)
_allocate_output(buffers::ProjectionBuffers) =
    QProjections(map(sz -> Matrix{Float64}(undef, sz...), buffers.sizes)...)

"""
    RSMWorkspace(geom, gridder_size; output=:projections)

Preallocated buffers for [`rss`](@ref)/[`rss!`](@ref)/[`rsm`](@ref)/[`rsm!`](@ref),
sized for a `geom`-shaped frame gridded onto `gridder_size` (a 2-tuple for RSS,
a 3-tuple for RSM). It holds every intermediate buffer, including the running
unnormalized grids. `output` picks which grids an RSM workspace serves, as for
[`rsm`](@ref).
"""
struct RSMWorkspace{D, S}
    gridder_size::NTuple{D, Int}  # bins per q-component
    buffers::S                    # GridBuffers or ProjectionBuffers: the output mode
    q_buffer::Matrix{Float64}     # (D, npix): rows are the kept q-components
end

function RSMWorkspace(geom::Geometry, gridder_size::NTuple{2, Integer})
    n = Int.(gridder_size)
    return RSMWorkspace(n, GridBuffers(n), Matrix{Float64}(undef, 2, npixels(geom)))
end

function RSMWorkspace(geom::Geometry, gridder_size::NTuple{3, Integer};
                      output::Union{Symbol, AbstractString}=:projections)
    n = Int.(gridder_size)
    buffers = if Symbol(output) === :volume
        GridBuffers(n)
    elseif Symbol(output) === :projections
        ProjectionBuffers(n)
    else
        throw(ArgumentError("output must be :projections or :volume; got $(repr(output))"))
    end
    return RSMWorkspace(n, buffers, Matrix{Float64}(undef, 3, npixels(geom)))
end

# The workspace `rss!`/`rsm!` need for existing output buffers.
_default_workspace(geom::Geometry, image::AbstractMatrix) = RSMWorkspace(geom, size(image))
_default_workspace(geom::Geometry, volume::AbstractArray{<:Any, 3}) =
    RSMWorkspace(geom, size(volume); output=:volume)
_default_workspace(geom::Geometry, p::QProjections) = RSMWorkspace(geom, _gridder_size(p))

_check_mode(::AbstractArray, ::GridBuffers) = nothing
_check_mode(::QProjections, ::ProjectionBuffers) = nothing
_check_mode(outputs, ::Any) =
    throw(ArgumentError("$(typeof(outputs)) outputs do not match the workspace's output mode"))

function _check_outputs(outputs, ws::RSMWorkspace)
    if _gridder_size(outputs) != ws.gridder_size
        throw(ArgumentError("output size $(_gridder_size(outputs)) does not match workspace gridder_size $(ws.gridder_size)"))
    end
    _check_mode(outputs, ws.buffers)
    return nothing
end

"""
    allocate_output(geom, gridder_size; output=:projections)

Allocate the output buffer(s) for `rss!` (a 2-tuple `gridder_size`) or `rsm!`
(a 3-tuple). `output=:volume` gives one `gridder_size`-shaped array,
`output=:projections` gives a [`QProjections`](@ref) of the three axis-pair
matrices.
"""
allocate_output(::Geometry, gridder_size::NTuple{2, Integer}) =
    Matrix{Float64}(undef, gridder_size...)

function allocate_output(::Geometry, gridder_size::NTuple{3, Integer};
                         output::Union{Symbol, AbstractString}=:projections)
    n = Int.(gridder_size)
    if Symbol(output) === :volume
        return Array{Float64, 3}(undef, n...)
    elseif Symbol(output) === :projections
        return QProjections(map(pair -> Matrix{Float64}(undef, n[pair[1]], n[pair[2]]),
                                _PROJECTION_PAIRS)...)
    else
        throw(ArgumentError("output must be :projections or :volume; got $(repr(output))"))
    end
end

# Bounds are a `(mins, maxs)` pair of per-component tuples internally, and the
# flat `(min₁, max₁, min₂, max₂, …)` tuple of the public API at the boundary.
function _split_bounds(bounds::NTuple{L, Real}, ::Val{D}) where {L, D}
    if L != 2D
        throw(ArgumentError("expected $(2D) bounds for a $D-component map, got $L"))
    end
    return (ntuple(d -> Float64(bounds[2d - 1]), Val(D)),
            ntuple(d -> Float64(bounds[2d]), Val(D)))
end

_flatten_bounds(mins::NTuple{D}, maxs::NTuple{D}) where {D} =
    ntuple(k -> isodd(k) ? mins[cld(k, 2)] : maxs[cld(k, 2)], Val(2D))

@inline _combine_bounds(a, b) = (min.(a[1], b[1]), max.(a[2], b[2]))

# Per-component extrema over a `(D, N)` `points` matrix.
function _point_bounds(points::AbstractMatrix, ::Val{D}, ntasks) where {D}
    tmapreduce(_combine_bounds, axes(points, 2); ntasks, scheduler=:static) do p
        c = ntuple(d -> @inbounds(points[d, p]), Val(D))
        (c, c)
    end
end

# One angle tuple per frame from either a tuple shared by every frame or a
# per-frame collection. A bare `Real` is a single-axis chain.
_frame_angles(angles::Tuple{Vararg{Real}}, nframes::Integer) = fill(angles, nframes)
_frame_angles(angles::Real, nframes::Integer) = fill((angles,), nframes)
function _frame_angles(angles, nframes::Integer)
    if length(angles) != nframes
        throw(ArgumentError("expected $nframes per-frame angles, got $(length(angles))"))
    end
    return [a isa Real ? (a,) : Tuple(a) for a in angles]
end

# Frame count implied by an angle argument, `nothing` when shared by every frame.
_angle_count(::Tuple{Vararg{Real}}) = nothing
_angle_count(::Real) = nothing
_angle_count(angles) = length(angles)

# Fill `q` with the per-pixel q-components of one frame.
function _frame_q!(q::AbstractMatrix, geom::Geometry, indices::NTuple{D, Int},
                   sample_angles::Tuple, detector_angles::Tuple, ntasks) where {D}
    ft = FrameTransform(geom, sample_angles, detector_angles)
    _compute_q!(q, geom, ft, indices; ntasks)
    return q
end

# Global q-extrema over all frames, one frame's q resident at a time. Needs
# only the geometry and angles, not the frame data.
function _q_bounds(q::AbstractMatrix, geom::Geometry, indices::NTuple{D, Int},
                        sample_angles::AbstractVector, detector_angles::AbstractVector,
                        ntasks) where {D}
    bounds = (ntuple(_ -> Inf, Val(D)), ntuple(_ -> -Inf, Val(D)))
    for (sa, da) in zip(sample_angles, detector_angles)
        points = _frame_q!(q, geom, indices, sa, da, ntasks)
        bounds = _combine_bounds(bounds, _point_bounds(points, Val(D), ntasks))
    end
    return bounds
end

"""
    q_bounds(geom; sample_angles, detector_angles, projection=nothing,
             nframes=nothing, ntasks=4)

The q-extent a scan will cover, as a flat `(qxmin, qxmax, qymin, qymax, qzmin,
qzmax)` tuple for the `bounds` argument of [`rsm`](@ref) and
[`RSMAccumulator`](@ref). Only the geometry and angles are needed, no frame
data. `sample_angles` and `detector_angles` are in degrees, either one tuple
shared by every frame or per-frame collections; `nframes` is only needed when
both are shared. Pass `projection` to get the 4-tuple for an [`rss`](@ref)
axis pair instead.
"""
function q_bounds(geom::Geometry; sample_angles, detector_angles,
                  projection::Union{Nothing, Tuple}=nothing,
                  nframes::Union{Nothing, Integer}=nothing, ntasks::Integer=4)
    indices = isnothing(projection) ? (1, 2, 3) : _proj_indices(projection)
    n = if isnothing(nframes)
        something(_angle_count(sample_angles), _angle_count(detector_angles), 1)
    else
        Int(nframes)
    end
    q = Matrix{Float64}(undef, length(indices), npixels(geom))
    mins, maxs = _q_bounds(q, geom, indices, _frame_angles(sample_angles, n),
                                _frame_angles(detector_angles, n), ntasks)
    return _flatten_bounds(mins, maxs)
end

"""
    RSMAccumulator(geom; bounds, gridder_size=(200, 200, 200), output=:projections,
                   fuzzy_width=nothing, ntasks=4)
    RSMAccumulator(geom; bounds, gridder_size=(500, 500), projection=(:qx, :qz), …)

A reciprocal space map built one frame at a time. Add frames with
[`push!`](@ref) or [`append!`](@ref) and read the map out with `sum(acc)` or
`sum!(outputs, acc)` at any point; summing does not disturb the accumulator.
This is the interface to use when the frames do not all fit in memory.

`bounds` is required and fixes the grid up front: a flat `(qxmin, qxmax, qymin,
qymax, qzmin, qzmax)` tuple, or a 4-tuple when `gridder_size` is a 2-tuple.
[`q_bounds`](@ref) computes it from the geometry and scan angles alone.
`output`, `gridder_size` and `fuzzy_width` mean what they do for [`rsm`](@ref)
and [`rss`](@ref).

```julia
bounds = q_bounds(geom; sample_angles=thetas, detector_angles=(γ, δ))
acc = RSMAccumulator(geom; bounds, gridder_size=(200, 200, 200))
for (frame, θ) in image_stream
    push!(acc, frame; sample_angles=θ, detector_angles=(γ, δ))
end
projections = sum(acc)
```
"""
@kwdef mutable struct RSMAccumulator{D, GE <: Geometry, W <: RSMWorkspace{D}, R}
    const geom::GE
    const workspace::W
    const params::GridParams{D}
    const task_ranges::R          # per task: last-dimension ranges or point chunks
    const indices::NTuple{D, Int}
    const ntasks::Int
    nframes::Int = 0
end

function RSMAccumulator(geom::Geometry, ws::RSMWorkspace{D}, indices::NTuple{D, Int},
                        mins::NTuple{D, Float64}, maxs::NTuple{D, Float64};
                        fuzzy_width::Union{Nothing, Tuple}=nothing,
                        ntasks::Integer=4) where {D}
    if size(ws.q_buffer) != (D, npixels(geom))
        throw(ArgumentError("workspace.q_buffer size $(size(ws.q_buffer)) does not match ($D, $(npixels(geom)))"))
    end

    n = ws.gridder_size
    buffers = ws.buffers
    if buffers isa GridBuffers
        task_ranges = index_chunks(1:n[D]; n=ntasks)
    else
        task_ranges = index_chunks(1:npixels(geom); n=ntasks)
        _ensure_task_buffers!(buffers, length(task_ranges))
    end
    params = _grid_params(n, mins, maxs, fuzzy_width)

    acc = RSMAccumulator(; geom, workspace=ws, params, task_ranges, indices, ntasks)
    _clear!(buffers, acc)
    return acc
end

function RSMAccumulator(geom::Geometry; bounds::Tuple,
                        gridder_size::Tuple{Vararg{Integer}}=(200, 200, 200),
                        output::Union{Symbol, AbstractString}=:projections,
                        projection=(:qx, :qz), kwargs...)
    if length(gridder_size) == 2
        ws = RSMWorkspace(geom, gridder_size)
        indices = _proj_indices(projection)
    else
        ws = RSMWorkspace(geom, gridder_size; output)
        indices = (1, 2, 3)
    end
    mins, maxs = _split_bounds(bounds, Val(length(indices)))
    return RSMAccumulator(geom, ws, indices, mins, maxs; kwargs...)
end

# The three accumulation phases, one method per buffers type.
function _clear!(buffers::GridBuffers, acc::RSMAccumulator)
    @tasks for rng in acc.task_ranges
        @set scheduler = :static
        _clear_grid!(buffers.grid, buffers.gridder.norm, rng)
    end
    return nothing
end

function _clear!(buffers::ProjectionBuffers, acc::RSMAccumulator)
    @tasks for t in 1:length(acc.task_ranges)
        @set scheduler = :static
        for k in 1:3
            fill!(buffers.grids[t][k], 0.0)
            fill!(buffers.norms[t][k], 0.0)
        end
    end
    return nothing
end

function _accumulate!(buffers::GridBuffers, acc::RSMAccumulator, frame)
    points = acc.workspace.q_buffer
    # A single range owns the whole last dimension and can reject nothing.
    use_last_dim_bins = length(acc.task_ranges) > 1
    if use_last_dim_bins
        _last_dim_bins!(buffers.gridder, points, acc.params; ntasks=acc.ntasks)
    end

    @tasks for rng in acc.task_ranges
        @set scheduler = :static
        _accumulate_grid!(buffers.grid, buffers.gridder, points, frame,
                          acc.params, rng, use_last_dim_bins)
    end
    return nothing
end

function _accumulate!(buffers::ProjectionBuffers, acc::RSMAccumulator, frame)
    points = acc.workspace.q_buffer
    @tasks for t in 1:length(acc.task_ranges)
        @set scheduler = :static
        _accumulate_projections!(buffers.grids[t], buffers.norms[t], _PROJECTION_PAIRS,
                             points, frame, acc.params, acc.task_ranges[t])
    end
    return nothing
end

function _normalize!(grid::AbstractArray, buffers::GridBuffers, acc::RSMAccumulator)
    @tasks for rng in acc.task_ranges
        @set scheduler = :static
        _normalize_grid!(grid, buffers.grid, buffers.gridder.norm, rng)
    end
    return nothing
end

function _normalize!(p::QProjections, buffers::ProjectionBuffers, acc::RSMAccumulator)
    for (k, grid) in enumerate(_grids(p))
        _sum_task_buffers!(grid, buffers, k, length(acc.task_ranges), acc.ntasks)
    end
    return nothing
end

# Sum every task's copy of grid `k` into `grid` and normalize in one pass.
function _sum_task_buffers!(grid, buffers::ProjectionBuffers, k::Integer,
                           nbuffers::Integer, ntasks::Integer)
    nums = [buffers.grids[t][k] for t in 1:nbuffers]
    dens = [buffers.norms[t][k] for t in 1:nbuffers]

    @tasks for rng in index_chunks(1:length(grid); n=ntasks)
        @set scheduler = :static

        @inbounds for i in rng
            num = 0.0
            den = 0.0
            for t in eachindex(nums)
                num += nums[t][i]
                den += dens[t][i]
            end
            grid[i] = ifelse(den > 1e-16, num / den, 0.0)
        end
    end
    return nothing
end

# Frame count of `frames`, whose leading dimensions must match `geom.data_shape`.
function _check_frames(geom::Geometry{N}, frames::AbstractArray) where {N}
    shape = geom.data_shape
    if ndims(frames) < N || size(frames)[1:N] != shape
        throw(ArgumentError("frame size $(size(frames)) does not start with geom.data_shape $shape"))
    end
    return length(frames) ÷ npixels(geom)
end

# Accumulate one frame against the q-vectors already in the workspace.
function _accumulate!(acc::RSMAccumulator, frame::AbstractVector)
    _accumulate!(acc.workspace.buffers, acc, frame)
    acc.nframes += 1
    return acc
end

"""
    push!(acc::RSMAccumulator, frame; sample_angles, detector_angles)

Add one detector `frame`, taken at the given angles (in degrees), to the map.
`frame` must match `geom.data_shape`.
"""
function Base.push!(acc::RSMAccumulator, frame::AbstractArray;
                    sample_angles, detector_angles)
    if _check_frames(acc.geom, frame) != 1
        throw(ArgumentError("push! takes a single frame; use append! for a stack"))
    end
    return append!(acc, reshape(frame, size(frame)..., 1); sample_angles, detector_angles)
end

"""
    append!(acc::RSMAccumulator, frames; sample_angles, detector_angles)

Add a `(geom.data_shape..., Nframes)` stack of frames to the map. The angles
are given in degrees, either as one angle tuple shared by every frame or as a
per-frame collection of length `Nframes`.
"""
function Base.append!(acc::RSMAccumulator, frames::AbstractArray;
                      sample_angles, detector_angles)
    nframes = _check_frames(acc.geom, frames)
    frames_flat = reshape(frames, npixels(acc.geom), nframes)
    sample_angles = _frame_angles(sample_angles, nframes)
    detector_angles = _frame_angles(detector_angles, nframes)

    for i in 1:nframes
        _frame_q!(acc.workspace.q_buffer, acc.geom, acc.indices, sample_angles[i],
                  detector_angles[i], acc.ntasks)
        _accumulate!(acc, @view frames_flat[:, i])
    end
    return acc
end

"""
    empty!(acc::RSMAccumulator)

Discard every frame accumulated so far, leaving the geometry and grid intact.
"""
function Base.empty!(acc::RSMAccumulator)
    _clear!(acc.workspace.buffers, acc)
    acc.nframes = 0
    return acc
end

Base.length(acc::RSMAccumulator) = acc.nframes

"""
    sum(acc::RSMAccumulator)
    sum!(outputs, acc::RSMAccumulator)

The map accumulated so far, normalized per bin, as `DimArray`s: one for
`rss`-shaped and `output=:volume` accumulators, a [`QProjections`](@ref) of
three for `output=:projections`.

`sum` allocates the output; `sum!` writes into `outputs`, which must have the
shape [`allocate_output`](@ref) would give. Neither disturbs the accumulator,
so more frames can be added afterwards.
"""
Base.sum(acc::RSMAccumulator) = sum!(_allocate_output(acc.workspace.buffers), acc)

function Base.sum!(outputs::Union{AbstractArray{Float64}, QProjections},
                   acc::RSMAccumulator)
    _check_outputs(outputs, acc.workspace)
    _normalize!(outputs, acc.workspace.buffers, acc)
    return _to_dimarray(outputs, acc)
end

# Output axis for row `r` of the materialized q matrix.
function _q_dim(acc::RSMAccumulator, r::Integer)
    p = acc.params
    return Q_DIMS[acc.indices[r]](_axis(p.mins[r], p.maxs[r], p.sizes[r]))
end

_to_dimarray(grid::AbstractArray{<:Any, D}, acc::RSMAccumulator{D}) where {D} =
    DimArray(grid, ntuple(r -> _q_dim(acc, r), Val(D)))

function _to_dimarray(p::QProjections, acc::RSMAccumulator)
    wrapped = map(_grids(p), _PROJECTION_PAIRS) do grid, (a, b)
        DimArray(grid, (_q_dim(acc, a), _q_dim(acc, b)))
    end
    return QProjections(wrapped...)
end

# Shared core of `rss!`/`rsm!`: accumulate every frame and return the
# accumulator for the caller to sum.
function _accumulate_frames!(indices::NTuple{D, Int}, frames::AbstractArray,
                          geom::Geometry, ws::RSMWorkspace{D}, sample_angles,
                          detector_angles, bounds, fuzzy_width,
                          ntasks::Integer) where {D}
    nframes = _check_frames(geom, frames)

    mins, maxs = if isnothing(bounds)
        _q_bounds(ws.q_buffer, geom, indices, _frame_angles(sample_angles, nframes),
                       _frame_angles(detector_angles, nframes), ntasks)
    else
        _split_bounds(bounds, Val(D))
    end
    acc = RSMAccumulator(geom, ws, indices, mins, maxs; fuzzy_width, ntasks)

    if isnothing(bounds) && nframes == 1
        # A single-frame bounds scan leaves that frame's q in the buffer.
        _accumulate!(acc, vec(frames))
    else
        append!(acc, frames; sample_angles, detector_angles)
    end
    return acc
end

"""
    rss(frame, geom; sample_angles, detector_angles,
        gridder_size=(500, 500), projection=(:qx, :qz),
        bounds=nothing, fuzzy_width=nothing,
        workspace=nothing) -> DimArray

Bin a single detector `frame` into a 2D q-space image, allocating the output.
Equivalent to [`allocate_output`](@ref) followed by [`rss!`](@ref).
`sample_angles` and `detector_angles` are specified in degrees.
"""
function rss(frame::AbstractArray, geom::Geometry;
             gridder_size::NTuple{2, Integer}=(500, 500), kwargs...)
    return rss!(allocate_output(geom, gridder_size), frame, geom; kwargs...)
end

"""
    rss!(image, frame, geom; sample_angles, detector_angles,
         projection=(:qx, :qz), bounds=nothing, fuzzy_width=nothing,
         workspace=nothing) -> DimArray

In-place RSS: write the 2D q-space image into `image` and return a
`DimArray` that wraps it. The gridder size is `size(image)`. `workspace`
defaults to a freshly allocated [`RSMWorkspace`](@ref); pass one explicitly
to reuse buffers across frames.

`image` must be an `(nx, ny)` `AbstractMatrix{Float64}`, and `frame` must match
`geom.data_shape`. See [`rss`](@ref) for the meaning of the remaining keyword
arguments. `sample_angles` and `detector_angles` are specified in degrees.
"""
function rss!(image::AbstractMatrix{Float64}, frame::AbstractArray, geom::Geometry;
              sample_angles,
              detector_angles,
              projection=(:qx, :qz),
              bounds::Union{Nothing, NTuple{4, Real}}=nothing,
              fuzzy_width::Union{Nothing, NTuple{2, Real}}=nothing,
              workspace::Union{Nothing, RSMWorkspace{2}}=nothing,
              ntasks::Integer=4)
    ws = isnothing(workspace) ? _default_workspace(geom, image) : workspace
    _check_outputs(image, ws)

    acc = _accumulate_frames!(_proj_indices(projection), frame, geom, ws, sample_angles,
                           detector_angles, bounds, fuzzy_width, ntasks)
    return sum!(image, acc)
end

function _rsm(frames::AbstractArray, geom::Geometry;
              gridder_size::NTuple{3, Integer}=(200, 200, 200),
              output::Union{Symbol, AbstractString}=:projections, kwargs...)
    return rsm!(allocate_output(geom, gridder_size; output), frames, geom; kwargs...)
end

"""
    rsm(frames, geom; sample_angles, detector_angles,
        gridder_size=(200, 200, 200), output=:projections, bounds=nothing,
        fuzzy_width=nothing, workspace=nothing)

Bin a stack of detector `frames` into q-space, allocating the output.
Equivalent to [`allocate_output`](@ref) followed by [`rsm!`](@ref).
`sample_angles` and `detector_angles` are specified in degrees.

`output=:projections` (the default) returns a [`QProjections`](@ref) of the
three axis-pair projections as `DimArray`s, computed without materializing the
volume. `output=:volume` returns the full 3D volume as one `DimArray`.
`gridder_size` is the bin count per q-axis either way.

A projection is *not* a sum over the volume's third axis: as in [`rss`](@ref),
the collapsed component is averaged away by the per-bin normalization. It
covers the same points as the volume, so a point out of range along the
collapsed axis is dropped from the projection too.
"""
rsm(args...; kwargs...) = _rsm(args...; kwargs...)

# The implementation is in a separate function so that we can call it from the
# PythonCall extension with a PyArray without running into infinite recursion.
function _rsm!(outputs::Union{AbstractArray{Float64, 3}, QProjections},
               frames::AbstractArray, geom::Geometry;
               sample_angles,
               detector_angles,
               bounds::Union{Nothing, NTuple{6, Real}}=nothing,
               fuzzy_width::Union{Nothing, NTuple{3, Real}}=nothing,
               workspace::Union{Nothing, RSMWorkspace{3}}=nothing,
               ntasks::Integer=4)
    ws = isnothing(workspace) ? _default_workspace(geom, outputs) : workspace
    _check_outputs(outputs, ws)

    acc = _accumulate_frames!((1, 2, 3), frames, geom, ws, sample_angles,
                           detector_angles, bounds, fuzzy_width, ntasks)
    return sum!(outputs, acc)
end

"""
    rsm!(outputs, frames, geom; sample_angles, detector_angles, bounds=nothing,
         fuzzy_width=nothing, workspace=nothing)

In-place RSM: accumulate every frame into `outputs` and return `DimArray`s
wrapping it. `outputs` is either a 3D array (the volume, indexed by
`(qx, qy, qz)`) or a [`QProjections`](@ref), which picks the `rsm` mode; the
gridder size follows from its size(s).

`frames` is `(geom.data_shape..., Nframes)`. `sample_angles` and
`detector_angles` are in degrees, either one tuple shared by every frame or a
per-frame collection of length `Nframes`.

`bounds` is a flat `(qxmin, qxmax, qymin, qymax, qzmin, qzmax)` tuple; when
omitted the q-extent is computed in a first pass over the angles. See
[`rss!`](@ref) for the remaining keyword arguments.
"""
rsm!(args...; kwargs...) = _rsm!(args...; kwargs...)
