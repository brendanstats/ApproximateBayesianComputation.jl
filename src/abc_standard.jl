"""
Internal ABC Function for dealing with 1D prior distributions
"""
function abc_standard1D(summaryStatistics::Any,
                        N::Int64,
                        threshold::Float64,
                        sample_prior::Function,
                        forward_model::Function,
                      compute_distance::Function)
    acceptedDraws = Array{Float64}(N)
    acceptedDistances = Array{Float64}(N)
    accepted = 0
    totalSamples = 0
    while accepted < N
        totalSamples += 1
        proposal = sample_prior()
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)
        if proposalDistance < threshold
            accepted += 1
            acceptedDraws[accepted] = proposal
            acceptedDistances[accepted] = proposalDistance
        end
    end
    return acceptedDraws, acceptedDistances, totalSamples
end

function abc_standard1D(summaryStatistics::Any,
                        N::Int64,
                        threshold::Array{Float64, 1},
                        sample_prior::Function,
                        forward_model::Function,
                        compute_distance::Function)
    acceptedDraws = Array{Float64}(N)
    acceptedDistances = Array{Float64}(N, length(threshold))
    accepted = 0
    totalSamples = 0
    while accepted < N
        totalSamples += 1
        proposal = sample_prior()
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)
        if all(proposalDistance .< threshold)
            accepted += 1
            acceptedDraws[accepted] = proposal
            acceptedDistances[accepted, :] = proposalDistance
        end
    end
    return acceptedDraws, acceptedDistances, totalSamples
end

"""
Internal ABC Function for dealing with multi-dimentional prior distributions
"""
function abc_standardMultiD(summaryStatistics::Any,
                            N::Int64,
                            d::Int64,
                            threshold::Float64,
                            sample_prior::Function,
                            forward_model::Function,
                            compute_distance::Function)
    acceptedDraws = Array{Float64}(N, d)
    acceptedDistances = Array{Float64}(N)
    accepted = 0
    totalSamples = 0
    while accepted < N
        totalSamples += 1
        proposal = sample_prior()
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)
        if proposalDistance < threshold
            accepted += 1
            acceptedDraws[accepted, :] = proposal
            acceptedDistances[accepted] = proposalDistance
        end
    end
    return acceptedDraws, acceptedDistances, totalSamples
end

function abc_standardMultiD(summaryStatistics::Any,
                            N::Int64,
                            d::Int64,
                            threshold::Array{Float64, 1},
                            sample_prior::Function,
                            forward_model::Function,
                            compute_distance::Function)
    acceptedDraws = Array{Float64}(N, d)
    acceptedDistances = Array{Float64}(N, length(threshold))
    accepted = 0
    totalSamples = 0
    while accepted < N
        totalSamples += 1
        proposal = sample_prior()
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)
        if all(proposalDistance .< threshold)
            accepted += 1
            acceptedDraws[accepted, :] = proposal
            acceptedDistances[accepted, :] = proposalDistance
        end
    end
    return acceptedDraws, acceptedDistances, totalSamples
end

"""
Standard ABC Algorithm

# Arguments
* `summaryStatistics::Any` summary statistics of observed data used in
distance comparison.
* `N::Int64` Number of samples to accept before termination
* `threshold::Float64 or ::Array{Float64, 1}` acceptance threshold
* `sample_prior::Function` Function to draw a sample from the prior distribution
 from either a univariate or multivariate distribution called by
 `sample_prior()`.
* `forward_model::Function` Function to simulate data based on draw from
parameter prior.  The call `forward_model(sample_prior())` should run.
.
* `compute_distance::Function` Function to compute distance between output of
the `forward_model` and the provided `summaryStatistics`. The call
`compute_distance(forward_model(sample_prior()), summaryStatistics)` should run.

# Value
Returns an Array corresponding to an accepted parameter values and an Array of
the associate distances.
"""
function abc_standard(summaryStatistics::Any,
                      N::Int64,
                      threshold::Float64,
                      sample_prior::Function,
                      forward_model::Function,
                      compute_distance::Function)
    proposal = sample_prior()
    if typeof(proposal) <: Array
        return abc_standardMultiD(summaryStatistics, N, length(proposal), threshold, sample_prior, forward_model, compute_distance)
    else
        return abc_standard1D(summaryStatistics, N, threshold, sample_prior, forward_model, compute_distance)
    end
end

function abc_standard(summaryStatistics::Any,
                      N::Int64,
                      threshold::Array{Float64, 1},
                      sample_prior::Function,
                      forward_model::Function,
                      compute_distance::Function)
    proposal = sample_prior()
    if typeof(proposal) <: Array
        return abc_standardMultiD(summaryStatistics, N, length(proposal), threshold, sample_prior, forward_model, compute_distance)
    else
        return abc_standard1D(summaryStatistics, N, threshold, sample_prior, forward_model, compute_distance)
    end
end
