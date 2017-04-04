using ApproximateBayesianComputation

data = rand(Distributions.Normal(2.0, 0.5), 100)
sample_prior = make_sample_prior(Distributions.Normal(0.0, 0.5))
forward_model(μ::Float64) =  rand(Distributions.Normal(μ, 0.5), 100)

summaryStatistics1 = mean(data)
threshold1 = 0.05
compute_distance1(x::Array{Float64, 1}, y::Float64) =  abs(mean(x) - y)


summaryStatistics2 = [mean(data), median(data)]
threshold2 = [0.05, 0.05]
compute_distance2(x::Array{Float64, 1}, y::Array{Float64, 1}) = abs([mean(x), median(x)] - y)

N = 200

@time out00 = abc_standard(summaryStatistics1, N, 0.5, sample_prior, forward_model, compute_distance1)

@time out0 = abc_standard(summaryStatistics1, N, 0.25, sample_prior, forward_model, compute_distance1)

@time out1 = abc_standard(summaryStatistics1, N, 0.05, sample_prior, forward_model, compute_distance1)

@time out11 = abc_standard(summaryStatistics1, N, 0.01, sample_prior, forward_model, compute_distance1)

@time out12 = abc_standard(summaryStatistics1, N, 0.005, sample_prior, forward_model, compute_distance1)

@time out2 = abc_standard(summaryStatistics2, N, threshold2, sample_prior, forward_model, compute_distance2)
