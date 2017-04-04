using ApproximateBayesianComputation
addprocs(2)
@everywhere importall ApproximateBayesianComputation

#Start Testing...
@time data = rand(Distributions.Normal(2.0, 0.5), 100)
summaryStatistics = mean(data)
nsteps = 5
numParticles = 200
initialSample = 1000

@everywhere sample_prior, density_prior = make_model_prior(Distributions.Normal(0.0, 0.5))
@everywhere sample_kernel, density_kernel = make_normal_kernel()
@everywhere kernel_sd = weightedstd2

@everywhere compute_distance(x::Array{Float64, 1}, y::Float64) = abs(mean(x) - y)
@everywhere forward_model(μ::Float64) = rand(Distributions.Normal(μ, 0.5), 100)

@everywhere rank_distances = identity
@everywhere shrink_threshold{A <: SingleMeasureAbcPmc}(abcpmc::A) = quantile_threshold(abcpmc, 0.3)

println("Testing ppmc_start")
@time step1 = ppmc_start(summaryStatistics, numParticles, initialSample, sample_prior,
                         compute_distance, forward_model, rank_distances)

println("Testing ppmc_step")
@time ppmc_step(step1, summaryStatistics, numParticles, kernel_sd, shrink_threshold, sample_kernel,
                forward_model, compute_distance, density_kernel, density_prior)

println("Testing pabc_pmc")
@time pout = pabc_pmc(summaryStatistics, nsteps, numParticles, initialSample, sample_prior,
                      density_prior, sample_kernel, density_kernel, compute_distance,
                      forward_model, rank_distances, kernel_sd, shrink_threshold)

@time out = abc_pmc(summaryStatistics, nsteps, numParticles, initialSample, sample_prior,
                    density_prior, sample_kernel, density_kernel, compute_distance,
                    forward_model, rank_distances, kernel_sd, shrink_threshold)
println(pout[1].threshold, out[1].threshold)
