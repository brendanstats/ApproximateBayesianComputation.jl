"""
Identical to `identity()` used for 1-d distances that do not need to be transformed
before being sorted
"""
function identiy_distance{G <: Real}(distances::Array{G, 1})
    return identity(distances)
end

function identiy_distance{G <: Real}(distances::Array{G, 2})
    return identity(distances)
end

"""
Function to return the permutation of the computed distances that will order from
smallest to largest.  In this case of multiple distance functions a cross dimention
comparison is made by summing the ranks for each dimention and then ordering the
sum from smallest to largest

# Arguments
* `distances::Array{Float64, 1}, SharedArray{Float64, 1}, Array{Array{Float64, 1}, 1}, SharedArray{Array{Float64, 1}, 1}`:
    vector of distances or vector of distance vectors

# Value
permutation of indicies so such that distances are ordered from smallest to largest
"""
function sumrank_distance{G <: Real}(distances::Array{G, 2})
    n, m = size(distances)
    rankArray = Array{Float64}(n, m)
    for jj in 1:m
        rankArray[:, jj] = sortperm(distances[:, jj])
    end
    totals = vec(sum(rankArray, 2))
    return totals
end


"""
Maximum rank seen in each row when ranks are compute for columns
`maxrank_distance(distances)`
A cross dimention comparison is made by ranking each observation across each
dimension and then assigning each observation the maximum across all dimentions.

# Arguments
* `distances::Array{Float64, 2}`: vector of distances or vector of distance vectors

# Value
Vector containing the maximum rank across each row

# Examples
`distances = rand(10,3)`
`maxrank_distance(distances)`
"""
function maxrank_distance{G <: Real}(distances::Array{G, 2})
    n, m = size(distances)
    rankArray = Array{Float64}(n, m)
    for jj in 1:m
        rankArray[:, jj] = sortperm(distances[:, jj])
    end
    totals = vec(maximum(rankArray, 2))
    return totals
end

"""
Normalized compound distance
`normrank_distance(distances)`

# Arguments
* `distances::Array{Float64, 2}`: vector of distances or vector of distance vectors

# Value
Vector containing sum across rows of normalized distances

# Examples
`distances = rand(10,3)`
`maxrank_distance(distances)`
"""
function normrank_distance{G <: Real}(distances::Array{G, 2})
    distmin = minimum(distances, 1)
    diststd = std(distances, 1)
    scaledDistances = (distances .- distmin) ./ diststd
    totals = vec(sum(scaledDistances))
    return totals
end

"""
Efficiently find the points across two distance metrics with exactly N entries <= both
"""
function minarea_distance{G <: Real}(x::Array{G, 1}, y::Array{G, 1}, N::Integer)
    permx = sortperm(x)
    rankx = invperm(permx)
    permy = sortperm(y)
    ranky = invperm(permy)
    yranks = falses(y)
    for ii in 1:(N - 1)
        yranks[ranky[permx[ii]]] = true
    end
    ymaxrank = length(yranks)
    
    xdist = x[permx[N:end]]
    ydist = zeros(xdist)
    ydist[1] = y[permy[ymaxrank]]
    for (ii, idx) in enumerate(permx[N:end])
        if ranky[idx] < ymaxrank
            yranks[ranky[idx]] = true
            ymaxrank = findprev(yranks, ymaxrank - 1)
        end
        ydist[ii] = y[permy[ymaxrank]]
    end
    return [xdist ydist]
end
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
