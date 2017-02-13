"""
Function to shrink threshold by weighted quantile of previously accepted distances

# Arguments
* `distances::Array{Float64, 1}`: vector of distances or vector of distance vectors
* `weights::WeightVec`: weight vector must be in WeightVec format
* `thresholdQuantile::Float64`: quantile (0,1) to shrink to

# Value
threshold acceptance thresholds
"""
function quantile_threshold{A <: SingleMeasureAbcPmc}(abcpmc::A, q::Float64)
    return StatsBase.quantile(abcpmc.distances, abcpmc.weights, q)
end

function independentquantile_threshold{A <: MultiMeasureAbcPmc}(abcpmc::A, q::Float64)
    m = size(abcpmc.distances, 2)
    thresholds = Array{Float64}(m)
    for jj in 1:m
        thresholds[jj] = quantile(abcpmc.distances[:, jj], abcpmc.weights, q)
    end
    return thresholds
end

function jointquantile_threshold{A <: MultiMeasureAbcPmc}(abcpmc::A, q::Float64)
    m = size(abcpmc.distances, 2)
    adjq = 1.0 - (1.0 - q) / m
    thresholds = Array{Float64}(m)
    for jj in 1:m
        thresholds[jj] = quantile(abcpmc.distances[:, jj], abcpmc.weights, adjq)
    end
    return thresholds
end
