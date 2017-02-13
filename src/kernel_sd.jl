"""
Calculate standard deviation used in transition kerenl, computes weighted standard
deviation on each set of particles individually
`wstd(pmcstep)`
`wstd(x, w)`
`wstd(A, w)`
"""
function weightedstd{A <: UnivariateAbcPmc}(abcpmc::A)
    return std(abcpmc.particles, abcpmc.weights)
end

function weightedstd{A <: MultivariateAbcPmc}(abcpmc::A)
    return vec(std(abcpmc.particles, abcpmc.weights, 1))
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
function weightedstd2{A <: UnivariateAbcPmc}(abcpmc::A)
    return sqrt(2) * std(abcpmc.particles, abcpmc.weights)
end

function weightedstd2{A <: MultivariateAbcPmc}(abcpmc::A)
    return sqrt(2) .* vec(std(abcpmc.particles, abcpmc.weights, 1))
end
