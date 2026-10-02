module QSpaceTools

export Geometry, rss, rss!, rsm, rsm!, RSMWorkspace, RSMAccumulator, QProjections,
       allocate_output, q_bounds

using DimensionalData: Dim, DimArray, AbstractDimArray, otherdims
using LinearAlgebra: normalize
using OhMyThreads: @tasks, @set, tmapreduce, index_chunks
using StaticArrays: SVector, SMatrix, @SMatrix
using PrecompileTools: @compile_workload

include("azimuthal_integration.jl")
include("geometry.jl")
include("gridder.jl")
include("rss.jl")

function _test_integrator(; shape, npt0, npt1, ndim)
    nbins = ndim == 2 ? npt0 * npt1 : npt0
    BakedIntegrator(
        ones(Int32, nbins + 1), Int32[], Float32[], Float32[],
        Float32.(1:npt0), Float32.(1:npt1),
        shape, "", "", "", npt0, npt1, ndim,
    )
end

@compile_workload begin
    images = rand(10, 10, 3)

    # 1D/2D integration
    b = _test_integrator(shape=(10, 10), npt0=10, npt1=0, ndim=1)
    integrate(b, images)
    b = _test_integrator(shape=(10, 10), npt0=10, npt1=4, ndim=2)
    integrate(b, images)

    # RSM/RSS functions
    sample_axes = ("y-", "x-", "z+")
    detector_axes = ("y-",)
    image_axes = ("z-", "y-")
    beam_direction = (1.0, 0.0, 0.0)
    sample_normal = (0.0, 0.0, 1.0)
    sample_faceup = "z+"
    distance = 3.14
    wavelength = 1.239
    pixel_size = 200e-6
    nch1, nch2 = size(images)[1:2]

    geom = Geometry(; sample_axes, detector_axes, image_axes,
                    beam_direction, sample_normal, sample_faceup,
                    pixel_size=(pixel_size, pixel_size),
                    center=(nch1 ÷ 2, nch2 ÷ 2),
                    shape=(nch1, nch2),
                    distance, wavelength)

    χ = -0.17
    ϕ = -0.004
    θ = 24.4
    twoθ = 47.75
    for output in (:volume, :projections)
        rsm(images, geom;
            gridder_size=(10, 10, 10),
            sample_angles=[(θ, χ, ϕ) for θ in 1.0:0.5:2.0],
            detector_angles=twoθ, output)
    end

    rss(images[:, :, 1], geom;
        sample_angles=(θ, χ, ϕ),
        detector_angles=(twoθ,))
end

end # module QSpaceTools
