"""
Implementation of Approximate Bayesian Computation (ABC) Algorithms

Includes standard ABC algorithm and Population Monte Carlo (PMC)variation.
Functions to make model definition easier are included as well as a variety
of methods for computing non parametric distances between data.
"""
module ApproximateBayesianComputation

import StatsBase, Distributions, JLD
import Base.copy

export ParticleDimension,
    Univariate,
    Multivariate,
    DistanceMeasure,
    SingleMeasure,
    MultiMeasure,
    AbcPmcStep,
    UnivariateAbcPmc,
    MultivariateAbcPmc,
    SingleMeasureAbcPmc,
    MultiMeasureAbcPmc,
    USAbcPmc,
    UMAbcPmc,
    MSAbcPmc,
    MMAbcPmc,
    AbcPmc
export kernel_weights, pmc_start, pmc_step, abc_pmc, abc_pmc_warmstart
export abc_standard, ABCResult, abc_standard1D, abc_standardMultiD
export boxcar_accept, gaussian_accept
export maxdist_bandwidth, gaussianprob_bandwidth
export weightedstd, weightedstd2
export make_normal_kernel, make_jointnormal_kernel, make_truncatednormal_kernel,
    make_joint_kernel
export make_sample_prior, make_density_prior, make_model_prior
export sample_particle_distance, ppmc_start, ppmc_step, pabc_pmc
export pabc_standard, find_particle, pabc_pmc_warmstart
export totalsamples_thresholds
export perm_distance, sumrank_distance, maxrank_distance, normrank_distance, minarea_distance
export quantile_acceptbw, independentquantile_acceptbw, jointquantile_acceptbw
export duration_to_string

include("AbcPmcStep.jl")
include("abc_pmc.jl")
include("abc_standard.jl")
include("accept_reject.jl")
include("initial_acceptbw.jl")
include("kernel_bandwidth.jl")
include("make_kernel_model.jl")
include("make_prior_model.jl")
include("parallel_abc_pmc.jl")
include("parallel_abc_standard.jl")
include("post_processing.jl")
include("rank_distances.jl")
include("shrink_acceptbw.jl")
include("utils.jl")

end # module
