using ApproximateBayesianComputation

#Start Testing...
data = rand(Distributions.Normal(2.0, 0.5), 100)

summaryStatistics = mean(data)
nsteps = 10
nparticles = 200
ninitial = 1000

sample_prior, density_prior = make_model_prior(Distributions.Normal(0.0, 0.5))
sample_kernel, density_kernel = make_normal_kernel()
kernel_bandwidth = weightedstd2

forward_model(μ::Float64) = rand(Distributions.Normal(μ, 0.5), 100)
compute_distance(x::Array{Float64, 1}, y::Float64) = abs(mean(x) - y)

rank_distances = identity
shrink_threshold{A <: SingleMeasureAbcPmc}(abcpmc::A) = quantile_threshold(abcpmc, 0.3)

out = abc_pmc(summaryStatistics, nsteps, nparticles, ninitial, sample_prior,
              density_prior, sample_kernel, density_kernel, forward_model,
              compute_distance, rank_distances, kernel_bandwidth, shrink_threshold)
