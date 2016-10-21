"""
Standard ABC Algorithm

# Arguments
* `summaryStatistics::Any` summary statistics of observed data used in
distance comparison.
* `N::Int64` Number of samples to accept before termination
* `threshold::Float64 or ::Array{Float64, 1}` acceptance threshold
* `sample_prior::Function` Function to draw a sample from the prior distribution.
* `forward_model::Function` Function to simulate data based on draw from parameter
prior.
* `compute_distance::Function` Function to compute distance between output of
the `forward_model` and the provided `summaryStatistics`.

# Value
Returns a 2D Array with each row corresponding to an accepted parameter value
Also returns the 2D array of associated distances
"""
function abc_standard(summaryStatistics::Any, N::Int64, threshold::Float64,
                      sample_prior::Function, forward_model::Function,
                      compute_distance::Function)
    exampleDraw = sample_prior()
    acceptedDraws = Array{typeof(exampleDraw[1])}(N, length(exampleDraw))
    acceptedDistances = Array{Float64}(N)
    accepted = 0
    while accepted < N
        proposal = sample_prior()
        simulatedStatistics = forward_model(proposal)
        proposalDistance = compute_distance(simulatedStatistics, summaryStatistics)
        if proposalDistance < threshold
            accepted += 1
            acceptedDraws[accepted, :] = proposal
            acceptedDistances[accepted] = proposalDistance
        end
    end
    return acceptedDraws, acceptedDistances
end

"""
Accepts with a Float64 or an Array{Float64, 1} as a threshold.
"""
function abc_standard(summaryStatistics::Any, N::Int64, threshold::Array{Float64, 1},
                      sample_prior::Function, forward_model::Function,
                      compute_distance::Function)
    exampleDraw = sample_prior()
    exampleStatistics = forward_model(exampleDraw)
    exampleDistance = compute_distance(exampleStatistics, summaryStatistics)
    acceptedDraws = Array{typeof(exampleDraw[1])}(N, length(exampleDraw))
    acceptedDistances = Array{Float64}(N, length(exampleDistance))
    accepted = 0
    while accepted < N
        proposal = sample_prior()
        simulatedStatistics = forward_model(proposal)
        proposalDistance = compute_distance(simulatedStatistics, summaryStatistics)
        if all(proposalDistance .< threshold)
            accepted += 1
            acceptedDraws[accepted, :] = proposal
            acceptedDistances[accepted, :] = proposalDistance
        end
    end
    return acceptedDraws, acceptedDistances
end
