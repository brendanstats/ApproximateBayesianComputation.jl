"""
Calculate standard deviation used in transition kerenl, computes weighted standard
deviation on each set of particles individually
`wstd(pmcstep)`
`wstd(x, w)`
`wstd(A, w)`
"""
function weightedstd(x::Array{G, 1}, w::W) where {G <: Real, W <: StatsBase.AnalyticWeights}
    return std(x, w, corrected = true)
end

function weightedstd(x::Array{G, 2}, w::W) where {G <: Real, W <: StatsBase.AnalyticWeights}
    return vec(std(x, w, 1, corrected = true))
end

function weightedstd(abcpmc::A) where A <: AbcPmcStep
    return weightedstd(abcpmc.particles, abcpmc.weights)
end

"""
Function
to calculate a variance used in the transition kernel.  Calculates the
weighted variance of particles and then multiples by 2.  In d-dimensional
case returns a vector of length d.

# Arguments
* `particles::Array{G <: Real, 1}`: accepted particles from previous ABC step
* `weights::WeightVec`: weights associated with particles

# Value
transition kernel variance, 2 * var(particles).  Calculated variance within particle types.
"""

function weightedstd2(x::Array{G, 1}, w::W) where {G <: Real, W <: StatsBase.AnalyticWeights}
    return sqrt(2.0) * std(x, w, corrected = true)
end

function weightedstd2(x::Array{G, 2}, w::W) where {G <: Real, W <: StatsBase.AnalyticWeights}
    return sqrt(2.0) .* vec(std(x, w, 1, corrected = true))
end

function weightedstd2(abcpmc::A) where A <: AbcPmcStep
    return weightedstd2(abcpmc.particles, abcpmc.weights)
end
