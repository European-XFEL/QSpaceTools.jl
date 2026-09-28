"""
Fractional-overlap N-dimensional histogramming, ported from xrayutilities'
`fuzzygridder2d`/`fuzzygridder3d`. Each input point is a finite box (default
width: half a bin) whose contribution is split across every bin it overlaps.
NaN data and out-of-range points are dropped.

Bin convention: with `n` bins on `[xmin, xmax]`, `dx = (xmax - xmin) / (n - 1)`
and `xmin`/`xmax` are the *centers* of the first and last bins, not the edges.
"""

# `unsafe_trunc(t + 0.5)` instead of `round` skips the banker's-rounding
# branch; the gridder's range guard keeps t ≥ 0.
@inline _bin_delta(xmin, xmax, n) = (xmax - xmin) / (n - 1)
@inline _bin_index(x, xmin, dx) = unsafe_trunc(Int, (x - xmin) / dx + 0.5)

@inline _axis(xmin, xmax, n) = range(Float64(xmin), Float64(xmax); length=Int(n))

# Caller-owned buffers: the `norm` denominator over the grid and the
# `_last_dim_bins!` table over the points, sized on demand.
struct GridderWorkspace{D}
    norm::Array{Float64, D}
    last_dim_lo::Vector{Int32}
    last_dim_hi::Vector{Int32}
end

function GridderWorkspace(gridder_size::NTuple{D, Integer}) where {D}
    return GridderWorkspace{D}(Array{Float64, D}(undef, Int.(gridder_size)...),
                               Int32[], Int32[])
end

# Per-axis grid parameters.
struct GridParams{D}
    mins::NTuple{D, Float64}
    maxs::NTuple{D, Float64}
    deltas::NTuple{D, Float64}
    widths::NTuple{D, Float64}
    dwidths::NTuple{D, Float64}
    sizes::NTuple{D, Int}
end

# Fraction of a box along dimension `d` that falls in bin `idx`, given the
# box's bin range `[lo, hi]` and center `val`.
@inline function _overlap(params::GridParams, d::Integer, idx, lo, hi, val)
    if lo == hi
        return 1.0
    end
    w_half = params.widths[d] / 2
    vmin, dv, dw = params.mins[d], params.deltas[d], params.dwidths[d]
    if idx == lo
        return (idx - (val - w_half - vmin + dv/2) / dv) / dw
    elseif idx == hi
        return ((val + w_half - vmin + dv/2) / dv - idx + 1) / dw
    else
        return 1.0 / dw
    end
end

# Per-point prologue shared by both accumulation kernels.
@inline _point_coords(points::AbstractMatrix, p::Integer, ::GridParams{D}) where {D} =
    ntuple(d -> points[d, p], Val(D))

@inline function _point_inside(coords::NTuple{D}, params::GridParams{D}) where {D}
    mins, maxs = params.mins, params.maxs
    return !any(ntuple(d -> coords[d] < mins[d] || coords[d] > maxs[d], Val(D)))
end

@inline function _point_bins(coords::NTuple{D}, params::GridParams{D}) where {D}
    mins, deltas, widths, sizes = params.mins, params.deltas, params.widths, params.sizes
    lo = ntuple(Val(D)) do d
        lower = coords[d] - widths[d] / 2
        lower <= mins[d] ? 1 : _bin_index(lower, mins[d], deltas[d]) + 1
    end
    hi = ntuple(Val(D)) do d
        upper = coords[d] + widths[d] / 2
        min(_bin_index(upper, mins[d], deltas[d]) + 1, sizes[d])
    end
    return (lo, hi)
end

function _check_gridder_inputs(image::AbstractArray{T, N}, ws::GridderWorkspace,
                               points::AbstractMatrix, data, bounds::Tuple,
                               widths::Union{Nothing, Tuple},
                               last_dim_range::Union{Nothing, UnitRange{Int}}) where {T, N}
    if size(image) != size(ws.norm)
        throw(DimensionMismatch("image $(size(image)) and norm $(size(ws.norm)) must match"))
    end
    if size(points, 1) != N
        throw(DimensionMismatch("points must have $N rows; got $(size(points, 1))"))
    end
    if size(points, 2) != length(data)
        throw(DimensionMismatch("points has $(size(points, 2)) columns but data has $(length(data)) entries"))
    end
    if length(bounds) != N
        throw(DimensionMismatch("bounds must have $N entries; got $(length(bounds))"))
    end
    if !isnothing(widths) && length(widths) != N
        throw(DimensionMismatch("widths must have $N entries; got $(length(widths))"))
    end

    if isnothing(last_dim_range)
        return 1:size(image, N)
    else
        if first(last_dim_range) < 1 || last(last_dim_range) > size(image, N)
            throw(ArgumentError("last_dim_range $last_dim_range escapes 1:$(size(image, N))"))
        end
        return last_dim_range
    end
end

# `sizes` is the bin count per grid dimension, or per q-component for
# `_accumulate_projections!`.
function _grid_params(sizes::NTuple{N, Integer}, mins::NTuple{N, Real},
                      maxs::NTuple{N, Real}, widths::Union{Nothing, Tuple}) where {N}
    n = map(Int, sizes)
    mins = map(Float64, mins)
    maxs = map(Float64, maxs)
    deltas = ntuple(d -> _bin_delta(mins[d], maxs[d], n[d]), Val(N))
    width_values = ntuple(Val(N)) do d
        isnothing(widths) || isnothing(widths[d]) ? deltas[d] / 2 : Float64(widths[d])
    end
    dwidths = ntuple(d -> width_values[d] / deltas[d], Val(N))
    return GridParams(mins, maxs, deltas, width_values, dwidths, n)
end

function _clear_grid!(image::AbstractArray{T, N}, norm::AbstractArray,
                      last_dim_range::UnitRange{Int}) where {T, N}
    ranges = ntuple(d -> d == N ? last_dim_range : axes(image, d), Val(N))
    image_zero = zero(eltype(image))
    norm_zero = zero(eltype(norm))
    @inbounds for I in CartesianIndices(ranges)
        image[I] = image_zero
        norm[I] = norm_zero
    end
    return nothing
end

# Precompute every point's bin range along the last dimension so tasks owning a range of it
# can reject points without the full prologue. Out-of-range points get an
# empty range.
function _last_dim_bins!(ws::GridderWorkspace, points::AbstractMatrix,
                         params::GridParams{D}; ntasks::Integer=1) where {D}
    last_dim_lo, last_dim_hi = ws.last_dim_lo, ws.last_dim_hi
    npoints = size(points, 2)
    if length(last_dim_lo) != npoints
        resize!(last_dim_lo, npoints)
        resize!(last_dim_hi, npoints)
    end
    mins, maxs, deltas, widths = params.mins, params.maxs, params.deltas, params.widths
    nlast = params.sizes[D]
    nt = ntasks
    @tasks for p in axes(points, 2)
        @set ntasks = nt
        @set scheduler = :static
        @inbounds begin
            c = points[D, p]
            if c < mins[D] || c > maxs[D]
                last_dim_lo[p], last_dim_hi[p] = Int32(1), Int32(0)
            else
                lower = c - widths[D] / 2
                l = lower <= mins[D] ? 1 : _bin_index(lower, mins[D], deltas[D]) + 1
                h = min(_bin_index(c + widths[D] / 2, mins[D], deltas[D]) + 1, nlast)
                last_dim_lo[p], last_dim_hi[p] = Int32(l), Int32(h)
            end
        end
    end
    return nothing
end

# `use_last_dim_bins` means `_last_dim_bins!` has run over these points; a call
# owning the whole last dimension can reject nothing, so it passes `false`.
function _accumulate_grid!(image::AbstractArray{T, N}, ws::GridderWorkspace,
                           points::AbstractMatrix, data,
                           params::GridParams, last_dim_range::UnitRange{Int},
                           use_last_dim_bins::Bool) where {T, N}
    norm = ws.norm
    last_dim_lo, last_dim_hi = ws.last_dim_lo, ws.last_dim_hi
    range_lo, range_hi = first(last_dim_range), last(last_dim_range)

    data_lin = vec(data)
    @inbounds for p in axes(points, 2)
        if use_last_dim_bins && (last_dim_hi[p] < range_lo || last_dim_lo[p] > range_hi)
            continue
        end

        v = data_lin[p]
        coords = _point_coords(points, p, params)
        if isnan(v) || !_point_inside(coords, params)
            continue
        end
        lo, hi = _point_bins(coords, params)

        # Box inside a single bin: every weight is 1.
        if all(ntuple(d -> lo[d] == hi[d], Val(N)))
            I = CartesianIndex(lo)
            image[I] += v
            norm[I] += one(eltype(norm))
            continue
        end

        # Clamp only the last dimension to the task's range; weights use the unclamped lo/hi.
        ranges = ntuple(Val(N)) do d
            d == N ? (max(lo[d], range_lo):min(hi[d], range_hi)) : (lo[d]:hi[d])
        end
        # Dimension 1 stays an ordinary inner loop for contiguous writes.
        for J in CartesianIndices(Base.tail(ranges))
            outer_weight = prod(ntuple(Val(N - 1)) do k
                d = k + 1
                _overlap(params, d, J[k], lo[d], hi[d], coords[d])
            end)
            for i in ranges[1]
                w = outer_weight * _overlap(params, 1, i, lo[1], hi[1], coords[1])
                image[i, J] += v * w
                norm[i, J] += w
            end
        end
    end
end

# One point's contribution to the 2D grid over q-components `a` and `b`.
@inline function _accumulate_2d!(grid::AbstractMatrix, norm::AbstractMatrix,
                                   v, coords, lo, hi, a::Int, b::Int,
                                   params::GridParams)
    @inbounds begin
        la, ha, lb, hb = lo[a], hi[a], lo[b], hi[b]
        if la == ha && lb == hb
            grid[la, lb] += v
            norm[la, lb] += one(eltype(norm))
            return nothing
        end

        for j in lb:hb
            wj = _overlap(params, b, j, lb, hb, coords[b])
            for i in la:ha
                w = wj * _overlap(params, a, i, la, ha, coords[a])
                grid[i, j] += v * w
                norm[i, j] += w
            end
        end
    end
    return nothing
end

# Accumulate the points in `prange` into all `G` 2D grids in one pass, `grids[k]`
# binning the component pair `pairs[k]`. `params` is per q-component, so the
# prologue is shared and every grid keeps exactly the points inside the full
# q-box. Callers parallelize over points, each task with its own `grids`/`norms`.
function _accumulate_projections!(grids::NTuple{G, AbstractMatrix},
                                  norms::NTuple{G, AbstractMatrix},
                                  pairs::NTuple{G, NTuple{2, Int}},
                                  points::AbstractMatrix, data,
                                  params::GridParams{D}, prange::UnitRange{Int}) where {G, D}
    data_lin = vec(data)
    @inbounds for p in prange
        v = data_lin[p]
        coords = _point_coords(points, p, params)
        if isnan(v) || !_point_inside(coords, params)
            continue
        end
        lo, hi = _point_bins(coords, params)

        map(grids, norms, pairs) do grid, norm, pair
            _accumulate_2d!(grid, norm, v, coords, lo, hi, pair[1], pair[2], params)
        end
    end
    return nothing
end

# `out` may alias `num`.
function _normalize_grid!(out::AbstractArray{<:Any, N}, num::AbstractArray,
                          norm::AbstractArray, last_dim_range::UnitRange{Int}) where {N}
    ranges = ntuple(d -> d == N ? last_dim_range : axes(out, d), Val(N))
    out_zero = zero(eltype(out))
    # 1e-16 floor copied from xrayutilities' gridder2d.c
    @inbounds for I in CartesianIndices(ranges)
        n = norm[I]
        out[I] = ifelse(n > 1e-16, num[I] / n, out_zero)
    end
end

"""
    fuzzygridder!(image, workspace, points, data, bounds;
                  widths=nothing, last_dim_range=nothing) -> image

Fractionally grid points in `N == ndims(image)` dimensions. `points` is
`(N, npoints)`, `bounds[d]` the `(min, max)` pair for dimension `d`, and
`widths[d]` the full box width (`nothing` entries default to half a bin).
`workspace` is a `GridderWorkspace` matching `size(image)`.

`image` is cleared before accumulation and normalized afterward. `last_dim_range`
restricts all writes to that range of the last dimension, allowing concurrent
calls on disjoint ranges. NaN entries of `data` are skipped.
"""
function fuzzygridder!(image::AbstractArray{T, N}, ws::GridderWorkspace,
                       points::AbstractMatrix, data, bounds::Tuple;
                       widths::Union{Nothing, Tuple}=nothing,
                       last_dim_range::Union{Nothing, UnitRange{Int}}=nothing) where {T, N}
    rng = _check_gridder_inputs(image, ws, points, data, bounds, widths, last_dim_range)
    params = _grid_params(size(image), map(first, bounds), map(last, bounds), widths)

    use_last_dim_bins = length(rng) != size(image, N)
    if use_last_dim_bins
        _last_dim_bins!(ws, points, params)
    end
    _clear_grid!(image, ws.norm, rng)
    _accumulate_grid!(image, ws, points, data, params, rng, use_last_dim_bins)
    _normalize_grid!(image, image, ws.norm, rng)

    return image
end
