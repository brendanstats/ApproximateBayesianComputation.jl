"""
Implementation of Approximate Bayesian Computation (ABC) Algorithms

Includes standard ABC algorithm and Population Monte Carlo (PMC)variation.
Functions to make model definition easier are included as well as a variety
of methods for computing non parametric distances between data.
"""
module ApproximateBayesianComputation

import StatsBase, Distributions, JLD
import Base.copy

export make_sample_prior, make_density_prior, make_model_prior
export make_normal_kernel, make_jointnormal_kernel, make_truncatednormal_kernel,
    make_joint_kernel
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

export weightedstd, weightedstd2
export perm_distance, sumrank_distance, maxrank_distance, normrank_distance
export quantile_threshold, independentquantile_threshold, jointquantile_threshold

export duration_to_string

export abc_standard, ABCResult
export kernel_weights, pmc_start, pmc_step, abc_pmc

export totalsamples_thresholds

include("make_prior_model.jl")
include("make_kernel_model.jl")
include("AbcPmcStep.jl")

include("kernel_sd.jl")
include("rank_distances.jl")
include("shrink_threshold.jl")
include("utils.jl")

include("abc_standard.jl")
include("abc_pmc.jl")

include("post_processing.jl")
end # module
