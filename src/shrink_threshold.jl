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

#function minarea_distance
#=
x = rand(1000)
y = rand(1000)

permx = sortperm(x)
permy = sortperm(y)

nless = Array{Int64}(1000, 1000)
x10 = Array{Float64}(0)
y10 = Array{Float64}(0)

@time for ii in 1:1000
    for jj in 1:1000
        nless[ii, jj] = sum(((x .<= x[ii]) .* (y .<= y[jj])))
        if nless[ii, jj] == 10
            push!(x10, x[ii])
            push!(y10, y[jj])
        end
    end
end

xyarea = x10 .* y10
extrema(xyarea)

permx[10]
permy[end]
nless[281, 412]

(x[permx[10]], y[permy[end]])

using RCall

R"plot($x, $y)"
R"points($x10, $y10, col = 'blue')"
R"hist($xyarea)"
R"boxplot($xyarea)"
=#
