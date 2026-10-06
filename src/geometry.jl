"""
Geometry layer for reciprocal-space conversion, mirroring xrayutilities'
`QConversion.area` + `init_area` for the HXRD case with UB = I. Each pixel
maps to `q = M_s^{-1} (M_d r̂_d - r̂_i) · 2π/λ` in the sample frame.

`Geometry` stores the lab-frame unit vector `r̂_d` of every pixel as a
`(3, npix)` array, normalized once at construction. Nothing downstream assumes
the pixels form a lattice, so multi-module detectors work too.
"""

const Vec3 = SVector{3, Float64}
const Mat3 = SMatrix{3, 3, Float64, 9}

# h·c in eV·Å, as in xrayutilities' `en2lam`.
const _HC_EV_ANGSTROM = 12398.419843320026

@inline energy2wavelength(energy_eV::Real) = _HC_EV_ANGSTROM / Float64(energy_eV)
@inline wavelength2energy(wavelength_Å::Real) = _HC_EV_ANGSTROM / Float64(wavelength_Å)

const AXIS_VECS = Dict(
    "x+" => Vec3(1.0, 0.0, 0.0),  "x-" => Vec3(-1.0,  0.0,  0.0),
    "y+" => Vec3(0.0, 1.0, 0.0),  "y-" => Vec3( 0.0, -1.0,  0.0),
    "z+" => Vec3(0.0, 0.0, 1.0),  "z-" => Vec3( 0.0,  0.0, -1.0),
)

parse_axis(v::AbstractVector) = Vec3(v)
parse_axis(v::Tuple) = Vec3(v)
function parse_axis(s::AbstractString)
    if !haskey(AXIS_VECS, s)
        throw(ArgumentError("Invalid axis string: $(repr(s))"))
    end

    AXIS_VECS[s]
end

# Right-handed rotation by `θ_deg` degrees around unit vector `e` (Rodrigues).
@inline function rotation_arb(θ_deg::Real, e::Vec3)
    s, c = sincosd(θ_deg)
    c1 = 1 - c
    ex, ey, ez = e

    @SMatrix [
        c + ex*ex*c1      ex*ey*c1 - ez*s   ex*ez*c1 + ey*s
        ey*ex*c1 + ez*s   c + ey*ey*c1      ey*ez*c1 - ex*s
        ez*ex*c1 - ey*s   ez*ey*c1 + ex*s   c + ez*ez*c1
    ]
end

"""
    Geometry

Geometry of an area-detector experiment, in xrayutilities conventions: the
sample/detector rotation chains, the incident beam, and one lab-frame unit
vector per pixel pointing from the sample to that pixel. `data_shape` is the
shape frames must have in their leading dimensions; the columns of
`directions` follow its column-major linear ordering.

`image_axes`, `pixel_size`, `center` and `distance` describe the (assembled)
2D image. The regular-grid constructor builds the pixel directions from them;
otherwise they are descriptive only and may be left `nothing`. Build a
`Geometry` with one of the two constructors below rather than directly.

`sample_normal` and `sample_faceup` are stored but unused: the q kernel
assumes UB = I.
"""
struct Geometry{N}
    sample_axes::Vector{Vec3}
    detector_axes::Vector{Vec3}
    image_axes::Union{Nothing, NTuple{2, Vec3}}
    beam_direction::Vec3
    sample_normal::Union{Nothing, Vec3}
    sample_faceup::Union{Nothing, Vec3}
    directions::Matrix{Float64}   # (3, npix) unit vectors, lab frame
    data_shape::Dims{N}
    pixel_size::Union{Nothing, NTuple{2, Float64}}
    center::Union{Nothing, NTuple{2, Float64}}
    distance::Union{Nothing, Float64}
    wavelength::Float64
end

npixels(g::Geometry) = size(g.directions, 2)

_show_vec(v::Vec3) = string("(", join(v, ", "), ")")
_show_axes(axes) = join((_show_vec(a) for a in axes), " ")
_show_axes(::Nothing) = "unset"

function Base.show(io::IO, ::MIME"text/plain", g::Geometry)
    println(io, "Geometry: ", join(g.data_shape, "×"), " (", npixels(g), " pixels)")
    println(io, "  sample axes    ", _show_axes(g.sample_axes))
    println(io, "  detector axes  ", _show_axes(g.detector_axes))
    println(io, "  image axes     ", _show_axes(g.image_axes))
    println(io, "  beam           ", _show_vec(g.beam_direction))
    println(io, "  pixel size     ", something(g.pixel_size, "unset"))
    println(io, "  center         ", something(g.center, "unset"))
    println(io, "  distance       ", something(g.distance, "unset"))
    print(io,   "  wavelength     ", g.wavelength)
end

# Optional metadata: convert when given, keep `nothing` otherwise.
_optional(f, x) = isnothing(x) ? nothing : f(x)
_float_pair(x) = (Float64(x[1]), Float64(x[2]))

function Base.show(io::IO, g::Geometry)
    print(io, "Geometry(", join(g.data_shape, "×"), ", λ=", g.wavelength, ")")
end

"""
    Geometry(positions, data_shape; sample_axes, detector_axes, beam_direction,
             wavelength, image_axes=nothing, sample_normal=nothing,
             sample_faceup=nothing, pixel_size=nothing, center=nothing,
             distance=nothing)

Build a geometry from explicit per-pixel `positions`: a `(3, npix)` array of
sample-to-pixel vectors in the lab frame, columns in the column-major order of
a `data_shape`-shaped frame. Only their directions matter, so any length unit
works. The optional arguments are metadata only.

This is the constructor for multi-module detectors; see the PythonCall
extension for building one from an EXtra-geom geometry.
"""
function Geometry(positions::AbstractMatrix{<:Real}, data_shape::NTuple{N, Integer};
                  sample_axes,
                  detector_axes,
                  beam_direction,
                  wavelength,
                  image_axes=nothing,
                  sample_normal=nothing,
                  sample_faceup=nothing,
                  pixel_size=nothing,
                  center=nothing,
                  distance=nothing) where {N}
    shape = Dims{N}(data_shape)
    npix = prod(shape)
    if size(positions) != (3, npix)
        throw(DimensionMismatch("positions must be (3, $npix) to match data_shape $shape; got $(size(positions))"))
    end

    directions = Matrix{Float64}(undef, 3, npix)
    @inbounds for k in 1:npix
        u = normalize(Vec3(positions[1, k], positions[2, k], positions[3, k]))
        directions[1, k] = u[1]
        directions[2, k] = u[2]
        directions[3, k] = u[3]
    end

    return Geometry{N}(
        [parse_axis(a) for a in sample_axes],
        [parse_axis(a) for a in detector_axes],
        _optional(ax -> (parse_axis(ax[1]), parse_axis(ax[2])), image_axes),
        parse_axis(beam_direction),
        _optional(parse_axis, sample_normal),
        _optional(parse_axis, sample_faceup),
        directions,
        shape,
        _optional(_float_pair, pixel_size),
        _optional(_float_pair, center),
        _optional(Float64, distance),
        Float64(wavelength),
    )
end

"""
    Geometry(; sample_axes, detector_axes, image_axes, beam_direction,
               pixel_size, center, shape, distance, wavelength,
               sample_normal=nothing, sample_faceup=nothing)

Build a geometry for a single detector laid out on a regular grid, in
xrayutilities conventions.

Axis fields accept either `"y-"`-style strings or 3-tuples / vectors.
`pixel_size` and `distance` must share a unit; nothing else depends on which.
`shape` is `(Nch1, Nch2)` — the same `(rows, cols)` order xrayutilities uses
for `init_area` — and `center` is `(cch1, cch2)`.
"""
function Geometry(;
        sample_axes,
        detector_axes,
        image_axes,
        beam_direction,
        pixel_size,
        center,
        shape,
        distance,
        wavelength,
        sample_normal=nothing,
        sample_faceup=nothing,
    )
    Nch1, Nch2 = Int(shape[1]), Int(shape[2])

    rpixel1 = Float64(pixel_size[1]) * parse_axis(image_axes[1])
    rpixel2 = Float64(pixel_size[2]) * parse_axis(image_axes[2])
    r_i_unit = normalize(parse_axis(beam_direction))
    # Lab-frame position of pixel (0, 0).
    r0 = Float64(distance) * r_i_unit -
         (Float64(center[1]) * rpixel1 + Float64(center[2]) * rpixel2)

    positions = Matrix{Float64}(undef, 3, Nch1 * Nch2)
    @inbounds for j2 in 0:(Nch2 - 1), j1 in 0:(Nch1 - 1)
        p = j1 * rpixel1 + j2 * rpixel2 + r0
        k = j2 * Nch1 + j1 + 1
        positions[1, k] = p[1]
        positions[2, k] = p[2]
        positions[3, k] = p[3]
    end

    return Geometry(positions, (Nch1, Nch2); sample_axes, detector_axes,
                    image_axes, beam_direction, sample_normal, sample_faceup,
                    pixel_size, center, distance, wavelength)
end

# Per-frame transform, independent of the pixel:
#     q = ms · (f · (md · rd_unit − r_i_unit))
#       = (f·ms·md) · rd_unit  +  (−f·ms·r_i_unit)
# so per pixel is one mat-vec plus one vec-add.
struct FrameTransform
    m_combined::Mat3    # f · ms · md
    q_offset::Vec3      # −f · ms · r_i_unit
end

function FrameTransform(g::Geometry, sample_angles, detector_angles)
    if length(sample_angles) != length(g.sample_axes)
        throw(ArgumentError("expected $(length(g.sample_axes)) sample angles, got $(length(sample_angles))"))
    end
    if length(detector_angles) != length(g.detector_axes)
        throw(ArgumentError("expected $(length(g.detector_axes)) detector angles, got $(length(detector_angles))"))
    end

    # ms = (R_0 · R_1 · … · R_{Ns-1})^{-1}
    ms_fwd = one(Mat3)
    for (ax, a) in zip(g.sample_axes, sample_angles)
        ms_fwd = ms_fwd * rotation_arb(Float64(a), ax)
    end
    ms = inv(ms_fwd)

    md = one(Mat3)
    for (ax, a) in zip(g.detector_axes, detector_angles)
        md = md * rotation_arb(Float64(a), ax)
    end

    f = 2π / g.wavelength
    r_i_unit = normalize(g.beam_direction)

    return FrameTransform(f * (ms * md), -f * (ms * r_i_unit))
end

# The rows `indices` of the transform, as a `D×3` matrix and a `D`-vector
function _project_transform(ft::FrameTransform, indices::NTuple{D, Int}) where {D}
    M = ft.m_combined
    # `k` walks column-major: row = mod1(k, D), col = cld(k, D).
    m_proj = SMatrix{D, 3, Float64, D * 3}(
        ntuple(k -> @inbounds(M[indices[mod1(k, D)], cld(k, D)]), Val(D * 3))
    )
    qo = SVector{D, Float64}(ntuple(k -> ft.q_offset[indices[k]], Val(D)))
    return m_proj, qo
end

# Per-component `(mins, maxs)` of the q-components `indices` over every pixel,
# without storing q.
function _q_extrema(g::Geometry, ft::FrameTransform, indices::NTuple{D, Int};
                    ntasks::Integer=4) where {D}
    m_proj, qo = _project_transform(ft, indices)
    dirs = g.directions

    tmapreduce(_combine_bounds, index_chunks(1:npixels(g); n=ntasks);
               scheduler=:static) do rng
        lo = SVector{D, Float64}(ntuple(_ -> Inf, Val(D)))
        hi = -lo
        @inbounds for k in rng
            v = m_proj * Vec3(dirs[1, k], dirs[2, k], dirs[3, k]) + qo
            lo = @fastmath min.(lo, v)
            hi = @fastmath max.(hi, v)
        end
        (Tuple(lo), Tuple(hi))
    end
end

# Fill `q_buffer` (`(D, npix)`) with the q-components `indices` of every
# pixel, projecting the transform to a `D×3` matrix once ahead of the loop.
function _compute_q!(q_buffer::AbstractMatrix{Float64},
                         g::Geometry, ft::FrameTransform,
                         indices::NTuple{D, Int}; ntasks::Integer=4) where {D}
    npix = npixels(g)
    if size(q_buffer) != (D, npix)
        throw(DimensionMismatch("workspace q_buffer $(size(q_buffer)) != ($D, $npix)"))
    end
    m_proj, qo = _project_transform(ft, indices)
    dirs = g.directions

    # Explicit inner loop per chunk: `@tasks` over `1:npix` would call the body
    # per element and block vectorization (2x slower).
    @tasks for rng in index_chunks(1:npix; n=ntasks)
        @set scheduler = :static

        @inbounds for k in rng
            d = Vec3(dirs[1, k], dirs[2, k], dirs[3, k])
            v = m_proj * d + qo
            ntuple(t -> (q_buffer[t, k] = v[t]; nothing), Val(D))
        end
    end
    return q_buffer
end

# Eager `(3, data_shape...)` array of per-pixel q-vectors; angles in degrees.
function pixel_q_array(g::Geometry, sample_angles, detector_angles)
    ft = FrameTransform(g, sample_angles, detector_angles)
    storage = Matrix{Float64}(undef, 3, npixels(g))
    _compute_q!(storage, g, ft, (1, 2, 3))
    return reshape(storage, 3, g.data_shape...)
end
