"""
Function to shrink acceptbw by weighted quantile of previously accepted distances

# Arguments
* `distances::Array{Float64, 1}`: vector of distances or vector of distance vectors
* `weights::WeightVec`: weight vector must be in WeightVec format
* `acceptbwQuantile::Float64`: quantile (0,1) to shrink to

# Value
acceptbw acceptance acceptbws
"""
function quantile_acceptbw{A <: SingleMeasureAbcPmc}(abcpmc::A, q::Float64)
    return StatsBase.quantile(abcpmc.distances, abcpmc.weights, q)
end

function independentquantile_acceptbw{A <: MultiMeasureAbcPmc}(abcpmc::A, q::Float64)
    m = size(abcpmc.distances, 2)
    acceptbws = Array{Float64}(m)
    for jj in 1:m
        acceptbws[jj] = quantile(abcpmc.distances[:, jj], abcpmc.weights, q)
    end
    return acceptbws
end

function jointquantile_acceptbw{A <: MultiMeasureAbcPmc}(abcpmc::A, q::Float64)
    m = size(abcpmc.distances, 2)
    adjq = 1.0 - (1.0 - q) / m
    acceptbws = Array{Float64}(m)
    for jj in 1:m
        acceptbws[jj] = quantile(abcpmc.distances[:, jj], abcpmc.weights, adjq)
    end
    return acceptbws
end
