"""
Define data structure to hold results of each iteration of ABC Population Monte Carlo
algorithm.

# Arguments
* `particles::Array` 1 or 2D array containing accepted particles
* `distances::Array` 1 or 2D array containing distances associated with accepted
 particles
* `weights::Array{Float64, 1}` array of weights associated with each accepted particle
* `threshold`
"""

#ABC Posterior can be over either a single parameter or multivarate
abstract ParticleDimension
type Univariate <: ParticleDimension end
type Multivariate <: ParticleDimension end

#Support posteriors based on both singe and multiple distance measures
abstract DistanceMeasure
type SingleMeasure <: DistanceMeasure end
type MultiMeasure <: DistanceMeasure end

#Base type for posterior object
abstract AbcPmcStep{P <: ParticleDimension, D <: DistanceMeasure}

#Define alias for easier function definition
typealias UnivariateAbcPmc{D <: DistanceMeasure} AbcPmcStep{Univariate, D}
typealias MultivariateAbcPmc{D <: DistanceMeasure} AbcPmcStep{Multivariate, D}

typealias SingleMeasureAbcPmc{P <: ParticleDimension} AbcPmcStep{P, SingleMeasure}
typealias MultiMeasureAbcPmc{P <: ParticleDimension} AbcPmcStep{P, MultiMeasure}

#Single parameter and single distance measure
type USAbcPmc{T <: Number, G <: Number} <: AbcPmcStep{Univariate, SingleMeasure}
    particles::Array{T, 1}
    distances::Array{G, 1}
    weights::StatsBase.WeightVec{Float64,Array{Float64,1}}
    threshold::G
    testedSamples::Array{Int64, 1}
end

#Single parameter and multiple distance measures
type UMAbcPmc{T <: Number, G <: Number} <: AbcPmcStep{Univariate, MultiMeasure}
    particles::Array{T, 1}
    distances::Array{G, 2}
    weights::StatsBase.WeightVec{Float64,Array{Float64,1}}
    threshold::Array{G, 1}
    testedSamples::Array{Int64, 1}
end

#Multiple parameters and single distance measure
type MSAbcPmc{T <: Number, G <: Number} <: AbcPmcStep{Multivariate, SingleMeasure}
    particles::Array{T, 2}
    distances::Array{G, 1}
    weights::StatsBase.WeightVec{Float64,Array{Float64,1}}
    threshold::G
    testedSamples::Array{Int64, 1}
end

#Multiple parameters and multiple distance measures
type MMAbcPmc{T <: Number, G <: Number} <: AbcPmcStep{Multivariate, MultiMeasure}
    particles::Array{T, 2}
    distances::Array{G, 2}
    weights::StatsBase.WeightVec{Float64,Array{Float64,1}}
    threshold::Array{G, 1}
    testedSamples::Array{Int64, 1}
end

#Constructor selecting type based on inputs
AbcPmc{T <: Number, G <: Number}(p::Array{T, 1}, d::Array{G, 1}, w::StatsBase.WeightVec{Float64,Array{Float64,1}}, t::G, ts::Array{Int64, 1}) = USAbcPmc(p, d, w, t, ts)
AbcPmc{T <: Number, G <: Number}(p::Array{T, 1}, d::Array{G, 2}, w::StatsBase.WeightVec{Float64,Array{Float64,1}}, t::Array{G, 1}, ts::Array{Int64, 1}) = UMAbcPmc(p, d, w, t, ts)
AbcPmc{T <: Number, G <: Number}(p::Array{T, 2}, d::Array{G, 1}, w::StatsBase.WeightVec{Float64,Array{Float64,1}}, t::G, ts::Array{Int64, 1}) = MSAbcPmc(p, d, w, t, ts)
AbcPmc{T <: Number, G <: Number}(p::Array{T, 2}, d::Array{G, 2}, w::StatsBase.WeightVec{Float64,Array{Float64,1}}, t::Array{G, 1}, ts::Array{Int64, 1}) = MMAbcPmc(p, d, w, t, ts)

#typeof(AbcPmc(randn(10), rand(10), StatsBase.WeightVec([fill(0.1, 5); fill(0.2, 5)]), 0.5, fill(5, 10)))
#typeof(AbcPmc(randn(10, 2), rand(10), StatsBase.WeightVec([fill(0.1, 5); fill(0.2, 5)]), 0.5, fill(5, 10)))
#typeof(AbcPmc(randn(10), rand(10, 2), StatsBase.WeightVec([fill(0.1, 5); fill(0.2, 5)]), [0.5, 0.5], fill(5, 10)))
#typeof(AbcPmc(randn(10, 2), rand(10, 2), StatsBase.WeightVec([fill(0.1, 5); fill(0.2, 5)]), [0.5, 0.5], fill(5, 10)))
