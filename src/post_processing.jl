#"""
#Functions to processes arrays for AbcPmcSteps
#"""

#{A <: UnivariateAbcPmc}
#{A <: MultivariateAbcPmc}
#{A <: SingleMeasureAbcPmc}
#{A <: MultiMeasureAbcPmc

#=
"""
Takes array of AbcPmcStep and returns an array with the first column the cumulative
samples drawn across steps and the second column the acceptance threshold used totalsamples_thresholds(x)
"""
=#
"""
test
"""
function totalsamples_thresholds{A <: SingleMeasureAbcPmc}(x::Array{A, 1})
    T = eltype(x[1].threshold)
    totalsamples = zeros(T, length(x))
    thresholds = zeros(T, length(x))
    for (ii, a) in enumerate(x)
        totalsamples[ii] = sum(T, a.testedSamples)
        thresholds[ii] = a.threshold
    end
    return (["samples", "threshold"], [cumsum(totalsamples) thresholds])
end

function totalsamples_thresholds{A <: MultiMeasureAbcPmc}(x::Array{A, 1})
    T = eltype(x[1].threshold)
    totalsamples = zeros(T, length(x))
    thresholds = zeros(T, length(x), length(x[1].threshold))
    for (ii, a) in enumerate(x)
        totalsamples[ii] = sum(T, a.testedSamples)
        thresholds[ii, :] = a.threshold
    end
    return (["samples"; map(x -> string("threshold", x), 1:size(thresholds, 2))], [cumsum(totalsamples) thresholds])
end
