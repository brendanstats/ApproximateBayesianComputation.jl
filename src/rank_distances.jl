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
function identiy_distance(distances::Array{Float64, 1})
    return identity(distances)
end


function sumrank_distance(distances::Array{Float64, 2})
    n, m = size(distances)
    rankArray = Array{Float64}(n, m)
    for jj in 1:m
        rankArray[:, jj] = sortperm(distances[:, jj])
    end
    totals = vec(sum(rankArray, 2))
    return totals
end


"""
Function to return the permutation of the computed distances that will order from
smallest to largest.  In this case of multiple distance functions a cross dimention
comparison is made by ranking each observation across each dimention and then assigning
each observation the maximum across all dimentions.  Ties are dealt with by using
the default value for sortperm().

# Arguments
* `distances::Array{Array{Float64, 1}, 1}, SharedArray{Array{Float64, 1}, 1}`:
    vector of distances or vector of distance vectors

# Value
permutation of indicies so such that distances are ordered from smallest to largest
"""
function maxrank_distance(distances::Array{Float64, 2})
    n, m = size(distances)
    rankArray = Array{Float64}(n, m)
    for jj in 1:m
        rankArray[:, jj] = sortperm(distances[:, jj])
    end
    totals = vec(maximum(rankArray, 2))
    return totals
end

function normrank_distance(distances::Array{Float64, 2})
    distmin = minimum(distances, 1)
    diststd = std(distances, 1)
    scaledDistances = (distances .- distmin) ./ diststd
    totals = vec(sum(scaledDistances))
    return totals
end
