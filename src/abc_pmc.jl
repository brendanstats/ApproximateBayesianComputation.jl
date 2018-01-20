"""
Function to calculate weights for new particles
    
# Arguments
* `newParticles`: particles from new time step of abcpmc algorithm
* `oldParticles`: particles from previous step of abcpmc algorithm
* `oldWeights::Weights`: weights from previous step of abcpmc algorithm
* `kernelbw`: variance used in transition kernel when drawing new particles
* `kernelDensity::Function`: density of transition kernel, takes a input value,
 old particle, and kernelbw
* `priorDensity::Function`: returns the prior density when given a particle

# Value
weights for new particles
"""
function kernel_weights{T <: Number, A <: UnivariateAbcPmc}(newParticles::Array{T, 1},
                                                            previousStep::A,
                                                            kernelbw::Float64,
                                                            density_kernel::Function,
                                                            density_prior::Function)
    if length(newParticles) != length(previousStep.particles)
        error("Length of new particles and old particles must match")
    end
    
    samplingDensity = zeros(Float64, size(newParticles, 1))
    for (p, w) in zip(previousStep.particles, previousStep.weights.values)
        samplingDensity += (density_kernel.(newParticles, p, kernelbw) .* w)
    end
    priorDensity = density_prior.(newParticles)
    newWeights = priorDensity ./ samplingDensity
    return StatsBase.AnalyticWeights(newWeights ./ sum(newWeights))
end

function kernel_weights{T <: Number, A <: MultivariateAbcPmc}(newParticles::Array{T, 2},
                                                              previousStep::A,
                                                              kernelbw::Array{Float64, 1},
                                                              density_kernel::Function,
                                                              density_prior::Function)
    if size(newParticles) != size(previousStep.particles)
        error("Length of new particles and old particles must match")
    end
    n = size(newParticles, 1)
    newWeights = zeros(Float64, n)
    for ii in 1:n
        samplingDensity = 0.0
        for (jj, w) in enumerate(previousStep.weights.values)
            samplingDensity += density_kernel(newParticles[ii, :],
                                                  previousStep.particles[jj, :],
                                                  kernelbw) * w
        end
        newWeights[ii] = density_prior(newParticles[ii, :]) / samplingDensity
    end
    return StatsBase.AnalyticWeights(newWeights ./ sum(newWeights))
end

"""
Initialize ABC PMC Algorithm
"""
function pmc_start(summaryStatistics::Any,
                   nparticles::Int64,
                   ninitial::Int64,
                   sample_prior::Function,
                   forward_model::Function,
                   compute_distance::Function,
                   rank_distances::Function,
                   initial_acceptbw::Function,
                   verbose::Bool = true)

    ##Draw first sample to determine typing
    proposal = sample_prior()
    simulatedData = forward_model(proposal)
    dist = compute_distance(simulatedData, summaryStatistics)

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
    for ii in 2:ninitial
        proposal = sample_prior()
        simulatedData = forward_model(proposal)
        particles[ii, :] = proposal
        distances[ii, :] = compute_distance(simulatedData, summaryStatistics)
    end
    
    ##Rank Particles
    if verbose println("Selecting Particles...") end
    ranks = rank_distances(distances)
    ordinals = sortperm(ranks)
    
    acceptbw = initial_acceptbw(distances, ordinals, nparticles)
    
    ##Subset Particles
    idxs = ordinals[1:nparticles]
    particles = subset(particles, idxs)
    distances = subset(distances, idxs)
    #=
    if typeof(proposal) <: Array
        particles = particles[idxs, :]
    else
        particles = particles[idxs]
    end

    if typeof(dist) <: Array
        distances = distances[idxs, :]
        acceptbw = vec(maximum(distances, 1))
    else
        distances = distances[idxs]
        acceptbw = maximum(distances)
    end
    =#

    ##Return result
    weights = StatsBase.AnalyticWeights(fill(1.0 / nparticles, nparticles))
    nsampled = fill(floor(Int64, ninitial / nparticles), nparticles)
    return AbcPmc(particles, distances, weights, acceptbw, nsampled)
end

"""
Function to new step in ABC Population Monte Carlo Algorithm
    
# Arguments
* `previousStep::ABCPMCStep` object containing information on previous step
* `shrink_acceptbw::Function` rule for computing reduced acceptance acceptbw
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
as computed distances, total number of parameters sampled, and acceptance acceptbw that
was used.
"""
function pmc_step{A <: AbcPmcStep}(previousStep::A, summaryStatistics::Any,
                                   nparticles::Int64, kernel_bandwidth::Function,
                                   shrink_acceptbw::Function,
                                   sample_kernel::Function,
                                   forward_model::Function,
                                   compute_distance::Function,
                                   density_kernel::Function,
                                   density_prior::Function,
                                   accept_reject::Function,
                                   verbose::Bool = true)
    
    #Allocate variables
    newParticles = zeros(previousStep.particles)
    newDistances = zeros(previousStep.distances)
    nsampled = zeros(previousStep.nsampled)

    #Compute kernel variance / std
    if verbose println("Calculating Bandwidth...") end
    kernelbw = kernel_bandwidth(previousStep)

    #Compute new acceptbw
    if verbose println("Calculating Acceptbw...") end
    acceptbw = shrink_acceptbw(previousStep)

    #Find new particles
    if verbose println("Sampling Particles...") end
    numAccepted = 0
    while numAccepted < nparticles
        nsampled[numAccepted + 1] += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelbw)

        #Simulate data
        simulatedData = forward_model(proposal)
        proposalDistance = compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        #if all(proposalDistance .< acceptbw)
        if accept_reject(proposalDistance, acceptbw)
            numAccepted += 1
            newParticles[numAccepted, :] = proposal
            newDistances[numAccepted, :] = proposalDistance
        end
    end
    
    if verbose  println("Calculating Weights...") end
    newWeights = kernel_weights(newParticles, previousStep, kernelbw,
                                density_kernel, density_prior)
    
    #Return result as AbcPmcStep
    return AbcPmc(newParticles, newDistances, newWeights, acceptbw, nsampled)
end

"""
Formatting for logging run setup
"""
function initlog(filename::String, startTime::DateTime, ninitial::Int64, nparticles::Int64, nsteps::Int64)
    logfile = open(filename, "w")
    write(logfile, "Run Info\n")
    write(logfile, "————————————————————","\n")
    write(logfile, string("Start Time: ", Dates.format(startTime, "Y-mm-dd HH:MM:SS"), "\n"))
    write(logfile, string("Initial Samples: ", ninitial, "\n"))
    write(logfile, string("Particles: ", nparticles, "\n"))
    write(logfile, string("Steps: ", nsteps, "\n"))
    write(logfile, "————————————————————","\n\n")
    close(logfile)
    nothing
end

"""
Formatting for logging step info
"""
function steplog{G <: AbstractFloat}(filename::String, nstep::Int64, totalTime::Base.Dates.Millisecond, stepTime::Base.Dates.Millisecond, stepacceptbw::G, nsampled::Int64)
    logfile = open(filename, "a")
        write(logfile, string("Step ", nstep, " Info\n"))
        write(logfile, "————————————————————","\n")
        write(logfile, string("Total Time: ", duration_to_string(totalTime), "\n"))
        write(logfile, string("Step Time: ", duration_to_string(stepTime), "\n"))
        write(logfile, string("Acceptbw: ", stepacceptbw, "\n"))
        write(logfile, string("Particles Sampled: ", nsampled, "\n"))
        write(logfile, "————————————————————","\n\n")
    close(logfile)
    nothing
end

function steplog{G <: AbstractFloat}(filename::String, nstep::Int64, totalTime::Base.Dates.Millisecond, stepTime::Base.Dates.Millisecond, stepacceptbw::Array{G, 1}, nsampled::Int64)
    logfile = open(filename, "a")
        write(logfile, string("Step ", nstep, " Info\n"))
        write(logfile, "————————————————————","\n")
        write(logfile, string("Total Time: ", duration_to_string(totalTime), "\n"))
        write(logfile, string("Step Time: ", duration_to_string(stepTime), "\n"))
        write(logfile, string("Acceptbw: ", stepacceptbw, "\n"))
        write(logfile, string("Particles Sampled: ", nsampled, "\n"))
        write(logfile, "————————————————————","\n\n")
    close(logfile)
    nothing
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
  choosing initial selection acceptbw
* `shrink_acceptbw::Function` method for shrinking acceptbw between iterations based on output
  from previous iteration
* `log::Bool = true` should progress be written to a log file
* `logFile::String = "log.txt"` file to write progress to, ignored if `log = false`
* `save::Bool = true` should results be written to a file
* `saveFile::String = "results.jld"` file to write results to
* `verbose::Bool = true` should progress be reported
"""
function abc_pmc(summaryStatistics::Any, nsteps::Int64,
                 nparticles::Int64, ninitial::Int64,
                 sample_prior::Function, density_prior::Function,
                 sample_kernel::Function, density_kernel::Function,
                 forward_model::Function, compute_distance::Function,
                 rank_distances::Function, kernel_bandwidth::Function,
                 shrink_acceptbw::Function;
                 initial_acceptbw::Function = maxdist_bandwidth,
                 accept_reject::Function = boxcar_accept,
                 verbose::Bool = true,
                 log::Bool = true, logFile::String = "log.txt",
                 save::Bool = true, saveFile::String = "results.jld")

    ##Write initial information is running log file
    if log
        startTime = now()
        initlog(logFile, startTime, ninitial, nparticles, nsteps)
    end

    ##Run first step
    results = [pmc_start(summaryStatistics,
                                  nparticles, ninitial,
                                  sample_prior, forward_model,
                                  compute_distance, rank_distances,
                                  initial_acceptbw, verbose)]
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
        push!(results, pmc_step(results[ii - 1], summaryStatistics, nparticles,
                                kernel_bandwidth, shrink_acceptbw, sample_kernel,
                                forward_model, compute_distance, density_kernel,
                                density_prior, accept_reject, verbose))

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

function abc_pmc_warmstart(nsteps::Int64, nparticles::Int64,
                           sample_prior::Function, density_prior::Function,
                           sample_kernel::Function, density_kernel::Function,
                           forward_model::Function, compute_distance::Function,
                           kernel_bandwidth::Function, shrink_acceptbw::Function;
                           accept_reject::Function = boxcar_accept,
                           verbose::Bool = true,
                           log::Bool = true, logFile::String = "log.txt",
                           save::Bool = true, saveFile::String = "results.jld")

    ##Print to log file    
    if verbose println("Resuming algorithm...") end

    ##Write initial information is running log file
    if log
        startTime = now()
    end

    ##Load results
    summaryStatistics = load(saveFile, "summaryStatistics")
    results = load(saveFile, "results")

    ##Check Number of steps
    if nsteps <= length(results)
        error("$nsteps alreadty complete")
    end
    
    #Subsequent steps
    for ii in length(results) + 1:nsteps
        push!(results, pmc_step(results[ii - 1], summaryStatistics, nparticles,
                                kernel_bandwidth, shrink_acceptbw, sample_kernel,
                                forward_model, compute_distance, density_kernel,
                                density_prior, accept_reject, verbose))

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
