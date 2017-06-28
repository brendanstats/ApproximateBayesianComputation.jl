"""
return maximum distance across each dimensions in the fist particles ranks
"""
function maxdist_bandwidth{G <: Real}(distances::Array{G, 1}, ordinals::Array{Int64, 1}, nparticles::Integer)
    return distances[ordinals[nparticles]]
end

function maxdist_bandwidth{G <: Real}(distances::Array{G, 2}, ordinals::Array{Int64, 1}, nparticles::Integer)
    return vec(maximum(subset(distances, ordinals[1:nparticles]), 1))
end

"""
Determine bandwidth based on 0.8 (default) acceptance rate for largest distance of accepted particle using a guassian acceptance kernel
"""
function gaussianprob_bandwidth{G <: Real}(distances::Array{G, 1}, ordinals::Array{Int64, 1}, nparticles::Integer; paccept::Float64 = 0.8)
    maxdist = maxdist_bandwidth(distances, ordinals, nparticles)
    return maxdist / sqrt(-2.0 * log(paccept))
end

function gaussianprob_bandwidth{G <: Real}(distances::Array{G, 2}, ordinals::Array{Int64, 1}, nparticles::Integer; paccept::Float64 = 0.8)
    maxdist = maxdist_bandwidth(distances, ordinals, nparticles)
    return maxdist ./ sqrt(-2.0 * log(paccept) / length(maxdist))
end
