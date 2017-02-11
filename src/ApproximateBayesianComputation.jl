"""
Implementation of Approximate Bayesian Computation (ABC) Algorithms

Includes standard ABC algorithm and Population Monte Carlo (PMC)variation.
Functions to make model definition easier are included as well as a variety
of methods for computing non parametric distances between data.
"""
module ApproximateBayesianComputation

import StatsBase, Distributions

export make_sample_prior, make_density_prior, make_model_prior
export abc_standard

include("make_prior_model.jl")
include("abc_standard.jl")

#include("abc_pmc.jl")


end # module
