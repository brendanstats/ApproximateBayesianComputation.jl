using ApproximateBayesianComputation, Base.Test

#Start Testing...
data = rand(Distributions.Normal(2.0, 0.5), 100)

summaryStatistics = mean(data)
nsteps = 2
numParticles = 100
initialSample = 1000
sample_prior, density_prior = make_model_prior(Distributions.Normal(0.0, 0.5))
sample_kernel, density_kernel = make_normal_kernel()
kernel_sd = weightedstd2
function compute_distance(x::Array{Float64, 1}, y::Float64)
    return abs(mean(x) - y)
end
function forward_model(μ::Float64)
    return rand(Distributions.Normal(μ, 0.5), 100)
end
rank_distances = identity
function shrink_threshold{A <: SingleMeasureAbcPmc}(abcpmc::A)
    return quantile_threshold(abcpmc, 0.3)
end
log = true
logFile = "log.txt"
save = true
saveFile = "results.jld"
verbose = true

@time out = abc_pmc(summaryStatistics, 15, numParticles, initialSample, sample_prior,
              density_prior, sample_kernel, density_kernel, compute_distance,
              forward_model, rank_distances, kernel_sd, shrink_threshold,
              verbose, log, logFile, save, saveFile)

out[8].threshold
out[10].threshold
out[12].threshold
out[15].threshold

using RCall
R"plot(density($(out[8].particles), weights = $(out[8].weights.values)))"
R"lines(density($(out[12].particles), weights = $(out[12].weights.values)), lty = 2)"
R"lines(density($(out[15].particles), weights = $(out[15].weights.values)), lty = 3)"
R"abline(v = $((sum(data) * 4.0) / (4.0 + 400.0)))"
