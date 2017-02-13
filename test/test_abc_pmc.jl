import StatsBase, Distributions, JLD
import Base.copy

include("../src/make_prior_model.jl")
include("../src/make_kernel_model.jl")
include("../src/AbcPmcStep.jl")

include("../src/kernel_sd.jl")
include("../src/rank_distances.jl")
include("../src/shrink_threshold.jl")
include("../src/utils.jl")

include("../src/abc_standard.jl")

function kernel_weights{T <: Number, A <: UnivariateAbcPmc}(newParticles::Array{T, 1},
                                                            previousStep::A,
                                                            kernelsd::Float64,
                                                            density_kernel::Function,
                                                            density_prior::Function)
    if length(newParticles) != length(previousStep.particles)
        error("Length of new particles and old particles must match")
    end
    
    samplingDensity = zeros(Float64, size(newParticles, 1))
    for (p, w) in zip(previousStep.particles, previousStep.weights.values)
        samplingDensity += (density_kernel.(newParticles, p, kernelsd) .* w)
    end
    priorDensity = density_prior.(newParticles)
    newWeights = priorDensity ./ sampleDensity
    return StatsBase.WeightVec(newWeights ./ sum(newWeights))
end

function pmc_step{A <: AbcPmcStep}(previousStep::A, summaryStatistics::Any,
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
    while numAccepted < numParticles
        testedSamples[numAccepted + 1] += 1

        #Draw Sample
        proposal = StatsBase.sample(previousStep)
        proposal = sample_kernel(proposal, kernelsd)

        #Simulate data
        simulatedData = forward_model(proposal)
        compute_distance(simulatedData, summaryStatistics)

        #Accept / reject
        if all(proposalDistance .< threshold) #depends on type
            numAccepted += 1
            newParticles[numAccepted, :] = proposal #depends on type
            newDistances[numAccepted, :] = proposalDistance #depends on type
        end
    end
    
    if verbose  println("Calculating Weights...") end
    newWeights = kernel_weights(newParticles, previousStep, kernelsd,
                                density_kernel, density_prior)
    
    #Return result as AbcPmcStep
    return AbcPmc(newParticles, newDistances, newWeights, threshold, testedSamples)
end


#Start Testing...
data = rand(Distributions.Normal(2.0, 0.5), 100)

summaryStatistics = mean(data)
nsteps = 2
numParticles = 100
initialSample = 1000
sample_prior, density_prior = make_model_prior(Distributions.Normal(0.0, 0.5))
sample_kernel, density_kernel = make_normal_kernel()
function compute_distance(x::Array{Float64, 1}, y::Float64)
    return abs(mean(x) - y)
end
function forward_model(μ::Float64)
    return rand(Distributions.Normal(μ, 0.5), 100)
end
rank_distances = identity
function shrink_threshold{A <: SingleMeasureAbcPmc}(abcpmc::A)
    return quantile_threshold(abcpmc, 0.9)
end
log = true
logFile = "log.txt"
save = true
saveFile = "results.jld"
verbose = true



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

##Determine types for particles and distances
sampleDraw = sample_prior()
if ndims(sampleDraw) == 0
    particles = Array{eltype(sampleDraw)}(initialSample)
elseif ndims(sampleDraw) == 1
    particles = Array{eltype(sampleDraw)}(initialSample, length(sampleDraw))
else
    error("sample_prior() should produce scalar or vector output")
end

sampleData = forward_model(sampleDraw)
sampleDistance = compute_distance(sampleData, summaryStatistics)
if ndims(sampleDistance) == 0
    distances = Array{typeof(sampleDistance)}(initialSample)
elseif ndims(sampleDistance) == 1
    distances = Array{typeof(sampleDistance)}(initialSample, length(sampleDistane))
else
    error("compute_distance should produce scalar or vector output")
end
particles[1, :] = sampleDraw
distances[1, :] = sampleDistance

##Calculate initial samples
if verbose println("Sampling Initial Particles and Computing Distances...") end
for ii in 2:initialSample
    proposal = sample_prior()
    simulatedData = forward_model(proposal)
    particles[ii, :] = proposal
    distances[ii, :] = compute_distance(simulatedData, summaryStatistics)
end

##Determine particles to accept
if verbose println("Selecting Particles...") end
ranks = rank_distances(distances)
sortedPermutation = sortperm(ranks)
particles = particles[sortedPermutation[1:numParticles], :]
distances = distances[sortedPermutation[1:numParticles], :]
weights = StatsBase.WeightVec(fill(1.0 / numParticles, numParticles))
threshold = maximum(distances, 1)
testedSamples = fill(floor(Int64, initialSample / numParticles), numParticles)
results = Dict(1 => AbcPmc(vec(particles), vec(distances), weights, threshold[1], testedSamples))

##Save results
if verbose println("Saving Step...") end
if save JLD.@save saveFile results end

##Print to log file    
if verbose println(string("Step: ", 1, " Complete")) end

stepTime = now() - startTime
totalTime = stepTime
if log
    logfile = open(logFile, "a")
    write(logfile, string("Step 1 Info\n"))
    write(logfile, "————————————————————","\n")
    write(logfile, string("Total Time: ", duration_to_string(totalTime)))
    write(logfile, string("Step Time: ", duration_to_string(totalTime)))
    write(logfile, string("Threshold: ", threshold, "\n"))
    write(logfile, string("Particles Tested: ", initialSample, "\n"))
    write(logfile, "————————————————————","\n\n")
    close(logfile)
end

#Subsequent steps
for ii in 2:nsteps
    results[ii] = pmc_step(results[ii - 1], summaryStatistics,
                           kernel_sd, shrink_threshold,
                           sample_kernel, forward_model,
                           compute_distance, density_kernel,
                           density_prior, verbose)

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
        write(logfile,string("Total Time: ", duration_to_string(totalTime)))
        write(logfile,string("Step Time: ", duration_to_string(totalTime)))
        write(file, string("Threshold: ", results[string("Step",ii)].threshold, "\n"))
        write(file, string("Particles Tested: ", results[string("Step",ii)].ntested, "\n"))
        write(logfile, "————————————————————","\n\n")
        close(logfile)
    end
    if verbose println(string("Step ", ii, " Complete")) end        
end

