"""
Sample a particle from the prior model and then return it and the associated distance
"""
function sample_particle_distance(summaryStatistics::Any, sample_prior::Function, forward_model::Function, compute_distance::Function)
    proposal = sample_prior()
    simulatedData = forward_model(proposal)
    return proposal, compute_distance(simulatedData, summaryStatistics)
end

function find_particle(summaryStatistics::Any, previousStep::A, threshold::G, kernelbandwidth::G, sample_kernel::Function, forward_model::Function, compute_distance::Function) where {A <: AbcPmcStep, G <: Real}

    sampled = 0
    while true
        sampled += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelbandwidth)

        #Simulate data
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        if proposalDistance < threshold
            return proposal, proposalDistance, sampled
        end
    end
end

function find_particle(summaryStatistics::Any, previousStep::A, threshold::Array{G, 1}, kernelbandwidth::G, sample_kernel::Function, forward_model::Function, compute_distance::Function) where {A <: AbcPmcStep, G <: Real}

    sampled = 0
    while true
        sampled += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelbandwidth)

        #Simulate data
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        if all(proposalDistance .< threshold)
            return proposal, proposalDistance, sampled
        end
    end
end

function find_particle(summaryStatistics::Any, previousStep::A, threshold::G, kernelbandwidth::Array{G, 1}, sample_kernel::Function, forward_model::Function, compute_distance::Function) where {A <: AbcPmcStep, G <: Real}

    sampled = 0
    while true
        sampled += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelbandwidth)

        #Simulate data
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        if proposalDistance < threshold
            return proposal, proposalDistance, sampled
        end
    end
end

function find_particle(summaryStatistics::Any, previousStep::A, threshold::Array{G, 1}, kernelbandwidth::Array{G, 1}, sample_kernel::Function, forward_model::Function, compute_distance::Function) where {A <: AbcPmcStep, G <: Real}

    sampled = 0
    while true
        sampled += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelbandwidth)

        #Simulate data
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        if all(proposalDistance .< threshold)
            return proposal, proposalDistance, sampled
        end
    end
end


"""
Initialize ABC PMC Algorithm
"""
function ppmc_start(summaryStatistics::Any,
                   nparticles::Int64, ninitial::Int64,
                   sample_prior::Function, forward_model::Function,
                   compute_distance::Function, rank_distances::Function,
                   verbose::Bool = true)

    ##Draw first sample to determine typing
    proposal, dist = sample_particle_distance(summaryStatistics, sample_prior, forward_model, compute_distance)

    ##Initialize Arrays based on draws
    if typeof(proposal) <: Array
        particles = Array{eltype(proposal)}(ninitial, length(proposal))
    else
        particles = Array{eltype(proposal)}(ninitial)
    end

    if typeof(dist) <: Array
        distances = Array{eltype(dist)}(ninitial, length(dist))
    else
        distances = Array{eltype(dist)}(ninitial)
    end
    
    particles[1, :] = proposal
    distances[1, :] = dist
    
    ##Calculate initial samples
    if verbose println("Sampling Initial Particles and Computing Distances...") end
    np = nprocs()
    ii = 1
    @sync begin
        for proc = 1:np
            if proc != myid() || np == 1
                @async begin
                    while true
                        idx = (idx = ii; ii += 1; idx)
                        if idx > ninitial
                            break
                        end
                        particles[idx, :], distances[idx, :] = remotecall_fetch(sample_particle_distance, proc, summaryStatistics, sample_prior, forward_model, compute_distance)
                    end
                end
            end
        end
    end
    
    ##Rank Particles
    if verbose println("Selecting Particles...") end
    ranks = rank_distances(distances)
    sortedPermutation = sortperm(ranks)

    ##Subset Particles
    idxs = sortedPermutation[1:nparticles]
    if typeof(proposal) <: Array
        particles = particles[idxs, :]
    else
        particles = particles[idxs]
    end

    if typeof(dist) <: Array
        distances = distances[idxs, :]
        threshold = vec(maximum(distances, 1))
    else
        distances = distances[idxs]
        threshold = maximum(distances)
    end

    ##Return result
    weights = StatsBase.AnalyticWeights(fill(1.0 / nparticles, nparticles))
    nsampled = fill(floor(Int64, ninitial / nparticles), nparticles)
    return AbcPmc(particles, distances, weights, threshold, nsampled)
end

"""
Function to new step in ABC Population Monte Carlo Algorithm
    
# Arguments
* `previousStep::ABCPMCStep` object containing information on previous step
* `shrink_threshold::Function` rule for computing reduced acceptance threshold
* `transition_kernel::Function` transition kernel for adding noise to previously
 accepted parameter values
* `forward_model::Function` simulates new data given parameter value
* `compute_distance::Function` calculates a distance measure between output of
 `forawrd_model` and supplied summary statistics
* `density_kernel::Function` density value for transition kernel returning
likelihood for new parameter values given old parameter values
* `density_prior::Function` evaluates density of parameter value for prior function
* `verbose::Bool = true` default value is true, should phase of step be reported
# Value
ABCPMCStep object containing new posterior distribution, parameters and weights, as well
as computed distances, total number of parameters sampled, and acceptance threshold that
was used.
"""
function ppmc_step(previousStep::A, summaryStatistics::Any, nparticles::Int64,
                  kernel_bandwidth::Function, shrink_threshold::Function,
                  sample_kernel::Function, forward_model::Function,
                  compute_distance::Function, density_kernel::Function,
                  density_prior::Function, verbose::Bool = true) where A <: AbcPmcStep
    
    #Allocate variables
    newParticles = zeros(previousStep.particles)
    newDistances = zeros(previousStep.distances)
    nsampled = zeros(previousStep.nsampled)

    #Compute kernel variance / std
    if verbose println("Calculating Bandwidth...") end
    kernelbandwidth = kernel_bandwidth(previousStep)

    #Compute new threshold
    if verbose println("Calculating Threshold...") end
    threshold = shrink_threshold(previousStep)

    #Find new particles
    if verbose println("Sampling Particles...") end
    accepted = 1
    np = nprocs()
    @sync begin
        for proc = 1:np
            if proc != myid() || np == 1
                @async begin
                    while true
                        idx = (idx = accepted; accepted += 1; idx)
                        if idx > nparticles
                            break
                        end
                        newParticles[idx, :], newDistances[idx, :], nsampled[idx] = remotecall_fetch(find_particle, proc, summaryStatistics, previousStep, threshold, kernelbandwidth, sample_kernel, forward_model, compute_distance)
                    end
                end
            end
        end
    end
        
    if verbose  println("Calculating Weights...") end
    newWeights = kernel_weights(newParticles, previousStep, kernelbandwidth,
                                density_kernel, density_prior)
    
    #Return result as AbcPmcStep
    return AbcPmc(newParticles, newDistances, newWeights, threshold, nsampled)
end

"""
Population Monte Carlo Approximate Bayesian Computation Algorithm

# Arguments
* `summaryStatistics` data for posterior to be conditioned on
* `steps::Int64` number of iterations to run
* `nparticles::Int64` number of posterior samples to return for each iteration
* `ninitial::Int64` number of samples to base initial iterationon
* `sample_prior::Function` generates a random draw from the prior
* `density_prior::Function` pdf of prior
* `compute_distance::Function` compute distance metric between output of
  `forward_model` and provided summary statistics
* `forward_model::Function` function to simulate summary statistics according to new
parameter value
* `transition_kernel::Function` transition kernel or "noise" function applied to particles
* `density_kernel::Function` pdf functions for transition kernel
* `rank_distances::Function` method for ranking particles based on computed distances and
  choosing initial selection threshold
* `shrink_threshold::Function` method for shrinking threshold between iterations based on output
  from previous iteration
* `log::Bool = true` should progress be written to a log file
* `logFile::String = "log.txt"` file to write progress to, ignored if `log = false`
* `save::Bool = true` should results be written to a file
* `saveFile::String = "results.jld"` file to write results to
* `verbose::Bool = true` should progress be reported
"""
function pabc_pmc(summaryStatistics::Any, nsteps::Int64,
                 nparticles::Int64, ninitial::Int64,
                 sample_prior::Function, density_prior::Function,
                 sample_kernel::Function, density_kernel::Function,
                 forward_model::Function, compute_distance::Function,
                 rank_distances::Function, kernel_bandwidth::Function,
                 shrink_threshold::Function; verbose::Bool = true,
                 log::Bool = true, logFile::String = "log.txt",
                 save::Bool = true, saveFile::String = "results.jld")

    ##Write initial information is running log file
    if log
        startTime = now()
        initlog(logFile, startTime, ninitial, nparticles, nsteps)
    end
    
    ##Run first step
    results = [ppmc_start(summaryStatistics,
                                  nparticles, ninitial,
                                  sample_prior, forward_model,
                                  compute_distance, rank_distances,
                                  verbose)]
    ##Save results
    if verbose println("Saving Step...") end
    if save JLD.@save saveFile summaryStatistics results end
    
    ##Print to log file    
    if verbose println(string("Step ", 1, " Complete")) end

    if log
        stepTime = now() - startTime
        totalTime = stepTime
        steplog(logFile, 1, totalTime, stepTime, results[1].acceptbw, ninitial)
    end
    
    #Subsequent steps
    for ii in 2:nsteps
        push!(results, ppmc_step(results[ii - 1], summaryStatistics, nparticles,
                                kernel_bandwidth, shrink_threshold, sample_kernel,
                                forward_model, compute_distance, density_kernel,
                                density_prior, verbose))

        ##Save results
        if verbose println("Saving Step...") end
        if save JLD.@save saveFile summaryStatistics results end

        #Print to log file
        if verbose println(string("Step ", ii, " Complete")) end

        if log
            stepTime = now() - startTime - totalTime
            totalTime += stepTime
            steplog(logFile, ii, totalTime, stepTime, results[ii].acceptbw, sum(results[ii].nsampled))
        end
    end
    return results
end

function pabc_pmc_warmstart(nsteps::Int64, nparticles::Int64,
                            sample_prior::Function, density_prior::Function,
                            sample_kernel::Function, density_kernel::Function,
                            forward_model::Function, compute_distance::Function,
                            kernel_bandwidth::Function, shrink_threshold::Function;
                            verbose::Bool = true,
                            log::Bool = true, logFile::String = "log.txt",
                            save::Bool = true, saveFile::String = "results.jld")

    ##Print to log file    
    if verbose println("Resuming algorithm...") end

    ##Write initial information is running log file
    if log
        startTime = now()
        totalTime = now() - startTime
    end
    
    ##Save results
    summaryStatistics = JLD.load(saveFile, "summaryStatistics")
    results = JLD.load(saveFile, "results")

    ##Check Number of steps
    if nsteps <= length(results)
        error("$nsteps alreadty complete")
    end
    
    #Subsequent steps
    for ii in length(results) + 1:nsteps
        push!(results, ppmc_step(results[ii - 1], summaryStatistics, nparticles,
                                kernel_bandwidth, shrink_threshold, sample_kernel,
                                forward_model, compute_distance, density_kernel,
                                density_prior, verbose))

        ##Save results
        if verbose println("Saving Step...") end
        if save JLD.@save saveFile results end

        #Print to log file
        if verbose println(string("Step ", ii, " Complete")) end

        if log
            stepTime = now() - startTime - totalTime
            totalTime += stepTime
            steplog(logFile, ii, totalTime, stepTime, results[ii].acceptbw, sum(results[ii].nsampled))
        end
    end
    return results
end
