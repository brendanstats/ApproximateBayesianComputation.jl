"""
Function in increase weighting of particles with realtively small distances
calculations done primarily in log space
"""
function distance_weights{A <: SingleMeasureAbcPmc}(abcpmc::A)
    logμ = log(mean(abcpmc.distances))
    lognewweight = (log(abcpmc.weights.values) - log(abcpmc.distances)) .+ logμ
    newweight = exp(lognewweight)
    return StatsBase.WeightVec(newweight ./ sum(newweight))
end

"""
Function in increase weighting of particles with realtively small distances
"""
function distancesq_weights{A <: SingleMeasureAbcPmc}(abcpmc::A)
    logμ = log(dot(abcpmc.distances, abcpmc.distances) / length(abcpmc.distances))
    lognewweight = (log(abcpmc.weights.values) - (2 .* log(abcpmc.distances))) .+ logμ
    newweight = exp(lognewweight)
    return StatsBase.WeightVec(newweight ./ sum(newweight))
end
