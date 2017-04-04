#"""
#Functions to processes arrays for AbcPmcSteps
#"""

#{A <: UnivariateAbcPmc}
#{A <: MultivariateAbcPmc}
#{A <: SingleMeasureAbcPmc}
#{A <: MultiMeasureAbcPmc


"""
`totalsamples_thresholds(Array{AbcPmcStep, 1})`

Takes array of AbcPmcStep and returns a tuple `(labels, [samples, thresholds])` where row `i` corresponds to the `ith` entry of the input Array.  The first column will list the total number of samples drawn in the corresponding step while subsequent columns will list threshold information.  In the case of a `MultiMeasureAbcPmc` array labels are given as 'threshold1', 'threshold2', ...
"""
function totalsamples_thresholds{A <: SingleMeasureAbcPmc}(x::Array{A, 1})
    T = eltype(x[1].threshold)
    totalsamples = zeros(T, length(x))
    thresholds = zeros(T, length(x))
    for (ii, a) in enumerate(x)
        totalsamples[ii] = sum(T, a.nsampled)
        thresholds[ii] = a.threshold
    end
    return (["samples", "threshold"], [cumsum(totalsamples) thresholds])
end

function totalsamples_thresholds{A <: MultiMeasureAbcPmc}(x::Array{A, 1})
    T = eltype(x[1].threshold)
    totalsamples = zeros(T, length(x))
    thresholds = zeros(T, length(x), length(x[1].threshold))
    for (ii, a) in enumerate(x)
        totalsamples[ii] = sum(T, a.nsampled)
        thresholds[ii, :] = a.threshold
    end
    return (["samples"; map(x -> string("threshold", x), 1:size(thresholds, 2))], [cumsum(totalsamples) thresholds])
end

"""
`particles_weights(Array{AbcPmcStep, 1})`

Takes array of AbcPmcStep and returns a tuple `(labels, [step, particles, weights])` where row `step` corresponds to the entry of the input Array.  In the case of a `MultivariateAbcPmc` array labels are given as 'particle1', 'particle2', ....  The last column contains the weights associated with the particles.
"""
function particles_weights{A <: UnivariateAbcPmc}(x::Array{A, 1})
    n = length(x)
    m = length(x[1].particles)
    out = Array{promote_type(eltype(x[1].particles), eltype(x[1].weights), Int64)}(n * m, 3)
    out[:, 1] = repeat(1:n, inner = m)
    for ii in 1:n
        rng = range((ii - 1) * m + 1, 1, m)
        out[rng, 2] = x[ii].particles
        out[rng, 3] = x[ii].weights.values
    end
    return (["step", "particle", "weight"], out)
end

function particles_weights{A <: MultivariateAbcPmc}(x::Array{A, 1})
    n = length(x)
    m, k = size(x[1].particles)
    out = Array{promote_type(eltype(x[1].particles), eltype(x[1].weights), Int64)}(n * m, k + 2)
    out[:, 1] = repeat(1:n, inner = m)
    for ii in 1:n
        rng = range((ii - 1) * m + 1, 1, m)
        out[rng, 2:(k + 1)] = x[ii].particles
        out[rng, k + 2] = x[ii].weights.values
    end
    return (["step"; map(x -> string("particle", x), 1:k); "weight"], out)
end

#workspace()
#whos()
