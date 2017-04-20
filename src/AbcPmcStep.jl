#=
Define data structure to hold results of each iteration of ABC Population Monte Carlo
algorithm.

# Arguments
* `particles::Array` 1 or 2D array containing accepted particles
* `distances::Array` 1 or 2D array containing distances associated with accepted
 particles
* `weights::Array{Float64, 1}` array of weights associated with each accepted particle
* `threshold`
* `nsampled`
=#

#ABC Posterior can be over either a single parameter or multivarate
"""
ABC PMC type describing number of parameters in distribution
###### Subtypes: `Univariate`, `Multivariate`
"""
abstract ParticleDimension

"""
ABC PMC type for single-parameter distributions
###### Supertype: `ParticleDimension`
"""
type Univariate <: ParticleDimension end

"""
ABC PMC type for multi-parameter distributions

###### Supertype: `ParticleDimension`
"""
type Multivariate <: ParticleDimension end

#Support posteriors based on both singe and multiple distance measures
"""
ABC PMC type describing number of distance metrics used
###### Subtypes: `SingleMeasure`, `MultiMeasure`
"""
abstract DistanceMeasure

"""
ABC PMC type describing single distance measure algorithms
###### Supertype: `DistanceMeasure`
"""
type SingleMeasure <: DistanceMeasure end

"""
ABC PMC type describing single distance measure algorithms
###### Supertype: `DistanceMeasure`
"""
type MultiMeasure <: DistanceMeasure end

#Base type for posterior object
"""
Define data structure to hold results of each iteration of ABC Population Monte Carlo
algorithm.

# Fieldnames
* `particles`accepted particles
* `distances::Array{G <: Real}` distances associated with accepted particles
* `weights::StatsBase.WeightVec{Float64, Array{Float64, 1}` importance weights each accepted particle
* `threshold::Array{G <: Real, 1} or G <: Real` acceptance threshold
* `nsampled::Array{Int64, 1}` number of samples tested before acceptance for each particle
"""
abstract AbcPmcStep{P <: ParticleDimension, D <: DistanceMeasure}

#Define alias for easier function definition
typealias UnivariateAbcPmc{D <: DistanceMeasure} AbcPmcStep{Univariate, D}
typealias MultivariateAbcPmc{D <: DistanceMeasure} AbcPmcStep{Multivariate, D}

typealias SingleMeasureAbcPmc{P <: ParticleDimension} AbcPmcStep{P, SingleMeasure}
typealias MultiMeasureAbcPmc{P <: ParticleDimension} AbcPmcStep{P, MultiMeasure}

#Single parameter and single distance measure
type USAbcPmc{T <: Number, G <: Real} <: AbcPmcStep{Univariate, SingleMeasure}
    particles::Array{T, 1}
    distances::Array{G, 1}
    weights::StatsBase.WeightVec{Float64,Array{Float64,1}}
    threshold::G
    nsampled::Array{Int64, 1}
end

#Single parameter and multiple distance measures
type UMAbcPmc{T <: Number, G <: Real} <: AbcPmcStep{Univariate, MultiMeasure}
    particles::Array{T, 1}
    distances::Array{G, 2}
    weights::StatsBase.WeightVec{Float64,Array{Float64,1}}
    threshold::Array{G, 1}
    nsampled::Array{Int64, 1}
end

#Multiple parameters and single distance measure
type MSAbcPmc{T <: Number, G <: Real} <: AbcPmcStep{Multivariate, SingleMeasure}
    particles::Array{T, 2}
    distances::Array{G, 1}
    weights::StatsBase.WeightVec{Float64,Array{Float64,1}}
    threshold::G
    nsampled::Array{Int64, 1}
end

#Multiple parameters and multiple distance measures
type MMAbcPmc{T <: Number, G <: Real} <: AbcPmcStep{Multivariate, MultiMeasure}
    particles::Array{T, 2}
    distances::Array{G, 2}
    weights::StatsBase.WeightVec{Float64,Array{Float64,1}}
    threshold::Array{G, 1}
    nsampled::Array{Int64, 1}
end

#Constructor selecting type based on inputs
AbcPmc{T <: Number, G <: Real}(p::Array{T, 1}, d::Array{G, 1}, w::StatsBase.WeightVec{Float64, Array{Float64,1}}, t::G, ts::Array{Int64, 1}) = USAbcPmc(p, d, w, t, ts)
AbcPmc{T <: Number, G <: Real}(p::Array{T, 1}, d::Array{G, 2}, w::StatsBase.WeightVec{Float64, Array{Float64,1}}, t::Array{G, 1}, ts::Array{Int64, 1}) = UMAbcPmc(p, d, w, t, ts)
AbcPmc{T <: Number, G <: Real}(p::Array{T, 2}, d::Array{G, 1}, w::StatsBase.WeightVec{Float64,Array{Float64,1}}, t::G, ts::Array{Int64, 1}) = MSAbcPmc(p, d, w, t, ts)
AbcPmc{T <: Number, G <: Real}(p::Array{T, 2}, d::Array{G, 2}, w::StatsBase.WeightVec{Float64, Array{Float64,1}}, t::Array{G, 1}, ts::Array{Int64, 1}) = MMAbcPmc(p, d, w, t, ts)

function copy{A <: AbcPmcStep}(abcpmc::A)
    return AbcPmc(copy(abcpmc.particles), copy(abcpmc.distances), StatsBase.WeightVec(abcpmc.weights.values), copy(abcpmc.threshold), copy(abcpmc.nsampled))
end

function StatsBase.sample{A <: UnivariateAbcPmc}(abcpmc::A)
    return StatsBase.sample(abcpmc.particles, abcpmc.weights)
end

function StatsBase.sample{A <: MultivariateAbcPmc}(abcpmc::A)
    row = StatsBase.sample(1:size(abcpmc.particles, 1), abcpmc.weights)
    return abcpmc.particles[row, :]
end
