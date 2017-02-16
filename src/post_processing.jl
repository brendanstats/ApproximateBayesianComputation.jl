"""
Functions to processes arrays for AbcPmcSteps
"""

#{A <: UnivariateAbcPmc}
#{A <: MultivariateAbcPmc}
#{A <: SingleMeasureAbcPmc}
#{A <: MultiMeasureAbcPmc

function totalsamples_thresholds{A <: SingleMeasureAbcPmc}(x::Array{A, 1})
    T = eltype(x[1].threshold)
    totalsamples = zeros(T, length(x))
    thresholds = zeros(T, length(x))
    for (ii, a) in enumerate(x)
        totalsamples[ii] = sum(T, a.testedSamples)
        thresholds[ii] = a.threshold
    end
    return [cumsum(totalsamples) thresholds]
end
