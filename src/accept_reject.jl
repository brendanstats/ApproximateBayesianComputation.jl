"""
return true / false based on hard thresholding of magnitudes above bandwidth
"""
function boxcar_accept{G <: Real}(proposalDistance::G, bandwidth::G)
    return abs(proposalDistance) < bandwidth
end

function boxcar_accept{G <: Real}(proposalDistance::Array{G, 1}, bandwidth::Array{G, 1})
    for (dist, bw) in zip(proposalDistance, bandwidth)
        if dist >= bw
            return false
        end
    end
    return true
end

"""
return true / false based on gaussian kernel acceptance probability
"""
function gaussian_accept{G <: Real}(proposalDistance::G, bandwidth::G)
    scaledDistane = proposalDistance / bandwidth
    return rand() < exp(-0.5 * scaledDistane ^ 2)
end

function gaussian_accept{G <: Real}(proposalDistance::Array{G, 1}, bandwidth::Array{G, 1})
    for (dist, bw) in zip(proposalDistance, bandwidth)
        scaledDistane = dist / bw
        if rand() >= exp(-0.5 * scaledDistane ^ 2)
            return false
        end
    end
    return true
end
