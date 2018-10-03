"""
return maximum distance across each dimensions in the fist particles ranks
"""
function maxdist_bandwidth(distances::Array{G, 1}, ordinals::Array{Int64, 1}, nparticles::Integer) where G <: Real
    return distances[ordinals[nparticles]]
end

function maxdist_bandwidth(distances::Array{G, 2}, ordinals::Array{Int64, 1}, nparticles::Integer) where G <: Real
    return vec(maximum(subset(distances, ordinals[1:nparticles]), 1))
end

"""
Determine bandwidth based on 0.8 (default) acceptance rate for largest distance of accepted particle using a guassian acceptance kernel
"""
function gaussianprob_bandwidth(distances::Array{G, 1}, ordinals::Array{Int64, 1}, nparticles::Integer; paccept::Float64 = 0.5) where G <: Real
    maxdist = maxdist_bandwidth(distances, ordinals, Int(nparticles / 2))
    return maxdist / sqrt(-2.0 * log(paccept))
end

function gaussianprob_bandwidth(distances::Array{G, 2}, ordinals::Array{Int64, 1}, nparticles::Integer; paccept::Float64 = 0.5) where G <: Real
    maxdist = maxdist_bandwidth(distances, ordinals, Int(nparticles / 2))
    return maxdist ./ sqrt(-2.0 * log(paccept) / length(maxdist))
end
