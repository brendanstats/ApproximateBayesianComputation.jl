"""
Sample a particle from the prior model and then return it and the associated distance
"""
function sample_particle_distance(summaryStatistics::Any, sample_prior::Function, forward_model::Function, compute_distance::Function)
    proposal = sample_prior()
    simulatedData = forward_model(proposal)
    return proposal, compute_distance(simulatedData, summaryStatistics)
end

function find_particle{A <: AbcPmcStep, G <: Real}(summaryStatistics::Any, previousStep::A, threshold::G, kernelsd::G, sample_kernel::Function, forward_model::Function, compute_distance::Function)

    tested = 0
    while true
        tested += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelsd)

        #Simulate data
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        if proposalDistance < threshold
            return proposal, proposalDistance, tested
        end
    end
end

function find_particle{A <: AbcPmcStep, G <: Real}(summaryStatistics::Any, previousStep::A, threshold::Array{G, 1}, kernelsd::G, sample_kernel::Function, forward_model::Function, compute_distance::Function)

    tested = 0
    while true
        tested += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelsd)

        #Simulate data
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        if all(proposalDistance .< threshold)
            return proposal, proposalDistance, tested
        end
    end
end

function find_particle{A <: AbcPmcStep, G <: Real}(summaryStatistics::Any, previousStep::A, threshold::G, kernelsd::Array{G, 1}, sample_kernel::Function, forward_model::Function, compute_distance::Function)

    tested = 0
    while true
        tested += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelsd)

        #Simulate data
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        if proposalDistance < threshold
            return proposal, proposalDistance, tested
        end
    end
end

function find_particle{A <: AbcPmcStep, G <: Real}(summaryStatistics::Any, previousStep::A, threshold::Array{G, 1}, kernelsd::Array{G, 1}, sample_kernel::Function, forward_model::Function, compute_distance::Function)

    tested = 0
    while true
        tested += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelsd)

        #Simulate data
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        if all(proposalDistance .< threshold)
            return proposal, proposalDistance, tested
        end
    end
end


"""
Initialize ABC PMC Algorithm
"""
function ppmc_start(summaryStatistics::Any,
                   numParticles::Int64, initialSample::Int64,
                   sample_prior::Function, forward_model::Function,
                   compute_distance::Function, rank_distances::Function,
                   verbose::Bool = true)

    ##Draw first sample to determine typing
    proposal, dist = sample_particle_distance(summaryStatistics, sample_prior, forward_model, compute_distance)

    ##Initialize Arrays based on draws
    if typeof(proposal) <: Array
        particles = Array{eltype(proposal)}(initialSample, length(proposal))
    else
        particles = Array{eltype(proposal)}(initialSample)
    end

    if typeof(dist) <: Array
        distances = Array{eltype(dist)}(initialSample, length(dist))
    else
        distances = Array{eltype(dist)}(initialSample)
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
                    while ii < initialSample
                        particles[ii, :], distances[ii, :] = remotecall_fetch(sample_particle_distance, proc, summaryStatistics, sample_prior, forward_model, compute_distance)
                        ii += 1
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
    idxs = sortedPermutation[1:numParticles]
    if typeof(proposal) <: Array
        particles = particles[idxs, :]
    else
        particles = particles[idxs]
    end

    if typeof(dist) <: Array
        distances = distances[idxs, :]
        threshold = maximum(distances, 1)
    else
        distances = distances[idxs]
        threshold = maximum(distances)
    end

    ##Return result
    weights = StatsBase.WeightVec(fill(1.0 / numParticles, numParticles))
    testedSamples = fill(floor(Int64, initialSample / numParticles), numParticles)
    return AbcPmc(particles, distances, weights, threshold, testedSamples)
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
function ppmc_step{A <: AbcPmcStep}(previousStep::A, summaryStatistics::Any, numParticles::Int64,
                  kernel_sd::Function, shrink_threshold::Function,
                  sample_kernel::Function, forward_model::Function,
                  compute_distance::Function, density_kernel::Function,
                  density_prior::Function, verbose::Bool = true)
    
    #Allocate variables
    newParticles = zeros(previousStep.particles)
    newDistances = zeros(previousStep.distances)
    testedSamples = zeros(previousStep.testedSamples)

    #Compute kernel variance / std
    if verbose println("Calculating Variance...") end
    kernelsd = kernel_sd(previousStep)

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
                    while accepted < numParticles
                        newParticles[accepted, :], newDistances[accepted, :], testedSamples[accepted] = remotecall_fetch(find_particle, proc, summaryStatistics, previousStep, threshold, kernelsd, sample_kernel, forward_model, compute_distance)
                        accepted += 1
                    end
                end
            end
        end
    end
        
    if verbose  println("Calculating Weights...") end
    newWeights = kernel_weights(newParticles, previousStep, kernelsd,
                                density_kernel, density_prior)
    
    #Return result as AbcPmcStep
    return AbcPmc(newParticles, newDistances, newWeights, threshold, testedSamples)
end

"""
Population Monte Carlo Approximate Bayesian Computation Algorithm

# Arguments
* `summaryStatistics` data for posterior to be conditioned on
* `steps::Int64` number of iterations to run
* `numParticles::Int64` number of posterior samples to return for each iteration
* `initialSample::Int64` number of samples to base initial iterationon
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
                 numParticles::Int64, initialSample::Int64,
                 sample_prior::Function, density_prior::Function,
                 sample_kernel::Function, density_kernel::Function,
                 forward_model::Function, compute_distance::Function,
                 rank_distances::Function, kernel_sd::Function,
                 shrink_threshold::Function, verbose::Bool = true,
                 log::Bool = true, logFile::String = "log.txt",
                 save::Bool = true, saveFile::String = "results.jld")

    ##Write initial information is running log file
    startTime = now()
    if log
        logfile = open(logFile, "a")
        write(logfile, "Run Info\n")
        write(logfile, "————————————————————","\n")
        write(logfile, string("Start Time:", Dates.format(startTime, "Y-mm-dd HH:MM:SS"), "\n"))
        write(logfile, string("Initial Samples:", initialSample, "\n"))
        write(logfile, string("Particles: ", numParticles, "\n"))
        write(logfile, string("Steps: ", nsteps, "\n"))
        write(logfile, "————————————————————","\n\n")
        close(logfile)
    end

    ##Run first step
    results = [ppmc_start(summaryStatistics,
                                  numParticles, initialSample,
                                  sample_prior, forward_model,
                                  compute_distance, rank_distances,
                                  verbose)]
    ##Save results
    if verbose println("Saving Step...") end
    if save JLD.@save saveFile results end

    ##Print to log file    
    if verbose println(string("Step ", 1, " Complete")) end

    stepTime = now() - startTime
    totalTime = stepTime
    if log
        logfile = open(logFile, "w")
        write(logfile, string("Step 1 Info\n"))
        write(logfile, "————————————————————","\n")
        write(logfile, string("Total Time: ", duration_to_string(totalTime), "\n"))
        write(logfile, string("Step Time: ", duration_to_string(totalTime), "\n"))
        write(logfile, string("Threshold: ", results[1].threshold, "\n"))
        write(logfile, string("Particles Tested: ", initialSample, "\n"))
        write(logfile, "————————————————————","\n\n")
        close(logfile)
    end

    #Subsequent steps
    for ii in 2:nsteps
        push!(results, ppmc_step(results[ii - 1], summaryStatistics, numParticles,
                                kernel_sd, shrink_threshold, sample_kernel,
                                forward_model, compute_distance, density_kernel,
                                density_prior, verbose))

        ##Save results
        if verbose println("Saving Step...") end
        if save JLD.@save saveFile results end

        #Print to log file
        stepTime = now() - startTime - totalTime
        totalTime += stepTime
        if log
            logfile = open(logFile, "a")
            write(logfile, string("Step ", ii, " Info\n"))
            write(logfile, "————————————————————","\n")
            write(logfile,string("Total Time: ", duration_to_string(totalTime), "\n"))
            write(logfile,string("Step Time: ", duration_to_string(stepTime), "\n"))
            write(logfile, string("Threshold: ", results[ii].threshold, "\n"))
            write(logfile, string("Particles Tested: ", results[ii].testedSamples, "\n"))
            write(logfile, "————————————————————","\n\n")
            close(logfile)
        end
        if verbose println(string("Step ", ii, " Complete")) end        
    end
    return results
end
