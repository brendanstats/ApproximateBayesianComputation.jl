"""
Function to shrink acceptbw by weighted quantile of previously accepted distances

# Arguments
* `distances::Array{Float64, 1}`: vector of distances or vector of distance vectors
* `weights::WeightVec`: weight vector must be in WeightVec format
* `acceptbwQuantile::Float64`: quantile (0,1) to shrink to

# Value
acceptbw acceptance acceptbws
"""
function quantile_acceptbw(d::Array{<:AbstractFloat, 1}, w::StatsBase.AnalyticWeights, q::AbstractFloat)
    return StatsBase.quantile(d, w, q)
end

function quantile_acceptbw{A <: SingleMeasureAbcPmc}(abcpmc::A, q::Float64)
    return StatsBase.quantile(abcpmc.distances, abcpmc.weights, q)
end

function independentquantile_acceptbw(d::Array{<:AbstractFloat, 2}, w::StatsBase.AnalyticWeights, q::AbstractFloat)
    m = size(d, 2)
    acceptbws = Array{Float64}(m)
    for jj in 1:m
        acceptbws[jj] = quantile(d[:, jj], w, q)
    end
    return acceptbws
end

function independentquantile_acceptbw(abcpmc::A, q::Float64) where A <: MultiMeasureAbcPmc
    return independentquantile_acceptbw(abcpmc.distances, abcpmc.weights, q)
end

function jointquantile_acceptbw(d::Array{<:AbstractFloat, 2}, w::StatsBase.AnalyticWeights, q::AbstractFloat)
    m = size(d, 2)
    return independentquantile_acceptbw(d, w, 1.0 - (1.0 - q) / m)
end

function jointquantile_acceptbw(abcpmc::A, q::Float64) where A <: MultiMeasureAbcPmc
    m = size(abcpmc.distances, 2)
    return independentquantile_acceptbw(abcpmc, 1.0 - (1.0 - q) / m)
end
