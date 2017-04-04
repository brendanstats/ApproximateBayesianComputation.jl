using ApproximateBayesianComputation
addprocs(2)
@everywhere importall ApproximateBayesianComputation

@time data = rand(Distributions.Normal(2.0, 0.5), 100)
summaryStatistics = mean(data)

@everywhere sample_prior1 = make_sample_prior(Distributions.Normal(0.0, 0.5))
@everywhere forward_model(μ::Float64) = rand(Distributions.Normal(μ, 0.5), 100)
@everywhere compute_distance(x::Array{Float64, 1}, y::Float64) = abs(mean(x) - y)
threshold = 0.05
N = 200

@time out = abc_standard(summaryStatistics, N, 0.1, sample_prior1, forward_model, compute_distance)
@time outp = pabc_standard(summaryStatistics, N, 0.1, sample_prior1, forward_model, compute_distance)
