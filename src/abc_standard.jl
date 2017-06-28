"""
Type for Basic ABC Output
# Arguments
* `particles`
* `distances`
* `acceptbw`
* `nsampled`
"""
type ABCResult{T <: Number, G <: Real}
    particles::Union{Array{T, 1}, Array{T, 2}}
    distances::Union{Array{G, 1}, Array{G, 2}}
    acceptbw::Union{G, Array{G, 1}}
    nsampled::Int64
end

"""
Standard ABC algorithm for univariate parameter of interest
"""
function abc_standard1D{G <: Real}(T::Type, summaryStatistics::Any,
                                   N::Int64,
                                   acceptbw::G,
                                   sample_prior::Function,
                                   forward_model::Function,
                                   compute_distance::Function,
                                   accept_reject::Function)
    #Allocate variables
    acceptedDraws = Array{T}(N)
    acceptedDistances = Array{G}(N)
    accepted = 0
    nsampled = 0

    #Repeat sampling until N particles accepted
    while accepted < N
        nsampled += 1

        #Draw sample and simulate data
        proposal = sample_prior()
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Retain sample is distance is below acceptbw
        #if proposalDistance < acceptbw
        if accept_reject(proposalDistance, acceptbw)
            accepted += 1
            acceptedDraws[accepted] = proposal
            acceptedDistances[accepted] = proposalDistance
        end
    end

    #Return result
    return ABCResult(acceptedDraws, acceptedDistances, acceptbw, nsampled)
end

function abc_standard1D{G <: Real}(T::Type, summaryStatistics::Any,
                                   N::Int64,
                                   acceptbw::Array{G, 1},
                                   sample_prior::Function,
                                   forward_model::Function,
                                   compute_distance::Function,
                                   accept_reject::Function)

    #Allocate variables
    acceptedDraws = Array{T}(N)
    acceptedDistances = Array{G}(N, length(acceptbw))
    accepted = 0
    nsampled = 0

    #Repeat sampling until N particles accepted
    while accepted < N
        nsampled += 1

        #Draw sample and simulate data
        proposal = sample_prior()
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Retain sample is distance is below acceptbw
        #if all(proposalDistance .< acceptbw)
        if accept_reject(proposalDistance, acceptbw)
            accepted += 1
            acceptedDraws[accepted] = proposal
            acceptedDistances[accepted, :] = proposalDistance
        end
    end

    #Return result
    return ABCResult(acceptedDraws, acceptedDistances, acceptbw, nsampled)
end

"""
Standard ABC algorithm for multivariate parameter of interest
"""
function abc_standardMultiD{G <: Real}(T::Type, summaryStatistics::Any,
                                       N::Int64,
                                       d::Int64,
                                       acceptbw::G,
                                       sample_prior::Function,
                                       forward_model::Function,
                                       compute_distance::Function,
                                       accept_reject::Function)

    #Allocate variables
    acceptedDraws = Array{T}(N, d)
    acceptedDistances = Array{G}(N)
    accepted = 0
    nsampled = 0

    #Repeat sampling until N particles accepted
    while accepted < N
        nsampled += 1

        #Draw sample and simulate data
        proposal = sample_prior()
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Retain sample is distance is below acceptbw
        #if proposalDistance < acceptbw
        if accept_reject(proposalDistance, acceptbw)
            accepted += 1
            acceptedDraws[accepted, :] = proposal
            acceptedDistances[accepted] = proposalDistance
        end
    end

    #Return result
    return ABCResult(acceptedDraws, acceptedDistances, acceptbw, nsampled)
end

function abc_standardMultiD{G <: Real}(T::Type, summaryStatistics::Any,
                                       N::Int64,
                                       d::Int64,
                                       acceptbw::Array{G, 1},
                                       sample_prior::Function,
                                       forward_model::Function,
                                       compute_distance::Function,
                                       accept_reject::Function)
    acceptedDraws = Array{T}(N, d)
    acceptedDistances = Array{G}(N, length(acceptbw))
    accepted = 0
    nsampled = 0

    #Repeat sampling until N particles accepted
    while accepted < N
        nsampled += 1

        #Draw sample and simulate data
        proposal = sample_prior()
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Retain sample is distance is below acceptbw
        #if all(proposalDistance .< acceptbw)
        if accept_reject(proposalDistance, acceptbw)
            accepted += 1
            acceptedDraws[accepted, :] = proposal
            acceptedDistances[accepted, :] = proposalDistance
        end
    end

    #Return result
    return ABCResult(acceptedDraws, acceptedDistances, acceptbw, nsampled)
end

"""
Standard ABC Algorithm, wrapper that determines dimention of algorithm to be called and runs appropriate version

# Arguments
* `summaryStatistics::Any` summary statistics of observed data used in
distance comparison.
* `N::Int64` Number of samples to accept before termination
* `acceptbw::Float64 or ::Array{Float64, 1}` acceptance acceptbw
* `sample_prior::Function` Function to draw a sample from the prior distribution from either a univariate or multivariate distribution called by `sample_prior()`.
* `forward_model::Function` Function to simulate data based on draw from parameter prior.  The call `forward_model(sample_prior())` should run.
* `compute_distance::Function` Function to compute distance between output of the `forward_model` and the provided `summaryStatistics`. The call `compute_distance(forward_model(sample_prior()), summaryStatistics)` should run.

# Value
Returns an object of type `ABCResult` containing the accepted particles, corresponding distances, acceptance acceptbw, and the number of samples drawn from the prior before the desired number of particles were accepted

# Examples
```julia
using ApproximateBayesianComputation

data = rand(Distributions.Normal(2.0, 0.5), 100)

#Construct Appropriate Functions
sample_prior = make_sample_prior(Distributions.Normal(0.0, 0.5))
function forward_model(μ::Float64)
    return rand(Distributions.Normal(μ, 0.5), 100)
end
function compute_distance(x::Array{Float64, 1}, y::Float64)
    return abs(mean(x) - y)
end

result = abc_standard(mean(data), 200, 0.05, sample_prior, forward_model, compute_distance, boxcar_accept)
```
"""
function abc_standard{G <: Real}(summaryStatistics::Any,
                                 N::Int64,
                                 acceptbw::Union{G, Array{G, 1}},
                                 sample_prior::Function,
                                 forward_model::Function,
                                 compute_distance::Function;
                                 accept_reject::Function = boxcar_accept)
    proposal = sample_prior()
    if typeof(proposal) <: Array
        return abc_standardMultiD(eltype(proposal), summaryStatistics, N, length(proposal), acceptbw, sample_prior, forward_model, compute_distance, accept_reject)
    else
        return abc_standard1D(typeof(proposal), summaryStatistics, N, acceptbw, sample_prior, forward_model, compute_distance, accept_reject)
    end
end
