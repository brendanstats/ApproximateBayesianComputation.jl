# ApproximateBayesianComputation

[![Build Status](https://travis-ci.org/brendanstats/ApproximateBayesianComputation.jl.svg?branch=master)](https://travis-ci.org/brendanstats/ApproximateBayesianComputation.jl)

[![Coverage Status](https://coveralls.io/repos/brendanstats/ApproximateBayesianComputation.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/brendanstats/ApproximateBayesianComputation.jl?branch=master)

[![codecov.io](http://codecov.io/github/brendanstats/ApproximateBayesianComputation.jl/coverage.svg?branch=master)](http://codecov.io/github/brendanstats/ApproximateBayesianComputation.jl?branch=master)

Implementation of standard rejection Approximate Bayesian Computation (ABC) algorithm as well the population Monte Carlo (PMC) described by (Beaumont et al. 2009) using importance weighting.  Additional sequential algorithms will be added in the future.

### Examples
Simple example estimating the posterior distribution of the mean of a normal distribution with a known variance using both the standard ABC rejection algorithm and 
```{julia}
using ApproximateBayesianComputation

data = rand(Distributions.Normal(2.0, 0.5), 100)
summaryStatistics = mean(data)

#Define functions for both standard and population Monte carlo algorithms
sample_prior, density_prior = make_model_prior(Distributions.Normal(0.0, 0.5))
sample_kernel, density_kernel = make_normal_kernel()
forward_model(μ::Float64) = rand(Distributions.Normal(μ, 0.5), 100)
compute_distance(x::Array{Float64, 1}, y::Float64) = abs(mean(x) - y)
rank_distances = identity
shrink_threshold{A <: SingleMeasureAbcPmc}(abcpmc::A) = quantile_threshold(abcpmc, 0.3)

#Other standard ABC parameters
threshold = 0.05
N = 200

out_standard = abc_standard(summaryStatistics,
             N,
             threshold,
             sample_prior,
             forward_model,
             compute_distance)

#Other PMC ABC parameters
numParticles = 200
initialSample = 1000
nsteps = 10

out_pmc = abc_pmc(summaryStatistics,
        nsteps,
        numParticles,
        initialSample,
        sample_prior,
        density_prior,
        sample_kernel,
        density_kernel,
        compute_distance,
        forward_model,
        rank_distances,
        kernel_sd,
        shrink_threshold)
```

### Algorithms
The standard rejection ABC algorithm is implemented vai the `abc_standard` function.  Following the algorithm:

    1. Draw a proposal θ from the prior π(θ)
    2. Simulate data x ∼ f(x|θ) using the forward model
    3. Accept θ is ρ(x, y) < ε where y is the supplied summary statistics, ρ is the supplied distance measure and ε is the supplied acceptance threshold
    4. Repeat 1. - 3. until the desired number of draws have been accepted.

The PMC ABC algorithm is implemented via the `abc_pmc` function.  Following the algorithm:

    1. Sample N₀ partilces θᵢ⁽¹⁾ from π(θ)
    2. For each particle simulate xᵢ⁽¹⁾ ∼ f(x|θᵢ⁽¹⁾)
    3. For each xᵢ⁽¹⁾ compute ρ(xᵢ⁽¹⁾, y) and retain the N {θ⁽¹⁾} associated with the smallest distances as determined by the selected `rank_dstances` function
    4. Compute importance weights wᵢ for each retained particle
    5. Resample from {θ⁽ᵗ⁾} according to weights {w⁽ᵗ⁾} and apply supplied transition kernel to generate θᵢ⁽ᵗ⁺¹⁾
    6. Compute distance ρ(xᵢ⁽ᵗ⁺¹⁾, y) where xᵢ⁽ᵗ⁺¹⁾ ∼ f(x|θᵢ⁽ᵗ⁺¹⁾)
    7. Accept θᵢ⁽ᵗ⁺¹⁾ is ρ(xᵢ⁽ᵗ⁺¹⁾, y) < ε⁽ᵗ⁺¹⁾ where ε⁽ᵗ⁺¹⁾ is computed based on the particles accepted in the previous step
    8. Generate new importance weights {w⁽ᵗ⁺¹⁾}
    9. Repeat 5. - 8. until N new particles have been accepted
    10. Repeat 9. for as many steps as specified.

### Prior and Kernel Construction
Function to add in the construction of function to sample and compute the density of the prior function are provided in the `make_sample_prior`, `make_density_prior`, `make_model_prior` functions.  These function take a distribution or `Array` for distributions as inputs and return functions for either sampling from the density or evaluating the density at a specified point.  In the case where an `Array` of distributions is supplied the dimensions are assumed to be independent.

Function for aiding in the construction of a transition kernel are also supplied as `make_normal_kernel`, `make_jointnormal_kernel`, `make_truncatednormal_kernel`, `make_joint_kernel`.  These assume that at each step a kernel bandwith parameter is computed by the `kerenl_sd` function and thus the generated sampling function take both a mean / centrality parameter and a bandwidth.  In the case of the generated density function a value at which the density is to be evaluated is also required.

### Threshold shrinkage and distance ranking
Several function are provided to shrink the acceptance threshold including `quantile_threshold`, `independentquantile_threshold`, `jointquantile_threshold`.  Similar the functions `sumrank_distance`, `maxrank_distance`, and `normrank_distance` are provided that allow for the mapping of a multidimensional distance function to a single dimension such that sorting can easily be achieved. 

### Parallelization
Parallel version of `abc_standard` and `abc_pmc` are provided in the `pabc_standard` and `pabc_pmc` functions.  All functions passed to these methods must be defined on the spawned workers, this can be accomplished via the `@everywhere` macro during initial definition.

### Types
Output for the standard algorithm is contained in `ABCResult` type which contains the fileds:

  * particles
  * distances
  * threshold
  * testedSamples

For the PMC ABC algorithm the output is provided in an `Array` of a subtype of the abstract type `AbcPmcStep{P <: ParticleDimension, D <: DistanceMeasure}`.  The `ParticleDimension` indicates either `Univariate` for single parameters or `Multivariate` for vector parameters.  `DistanceMeasure` takes the types `SingleMeasure` is a single distance measure is used or `MultiMeasure` if multiple distance measures are used.  Typealiases `UnivariateAbcPmc`, `MultivariateAbcPmc` are provided for defining methods affected by the number of parameters.  Similarly the typealiases `SingleMeasureAbcPmc` and `MultiMeasureAbcPmc` are provided to make method definition easier for methods affected by the dimension of the distance measures.  The four defined composite types are `USAbcPmc`, `MSAbcPmc`, `UMAbcPmc`, `MMAbcPmc` with each containing the fields:

  * particles
  * distances
  * weights
  * threshold
  * testedSamples

The weights are contained in a `StatsBase.WeightVec`