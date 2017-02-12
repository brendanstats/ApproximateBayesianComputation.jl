using ApproximateBayesianComputation, Base.Test, RCall

data = rand(Distributions.Normal(2.0, 0.5), 100)

sample_prior = make_sample_prior(Distributions.Normal(0.0, 0.5))
function forward_model(μ::Float64)
    return rand(Distributions.Normal(μ, 0.5), 100)
end

summaryStatistics1 = mean(data)
threshold1 = 0.05
function compute_distance1(x::Array{Float64, 1}, y::Float64)
    return abs(mean(x) - y)
end

summaryStatistics2 = [mean(data), median(data)]
threshold2 = [0.05, 0.05]
function compute_distance2(x::Array{Float64, 1}, y::Array{Float64, 1})
    return abs([mean(x), median(x)] - y)
end

N = 200

@time out1 = abc_standard(summaryStatistics1, N, threshold1, sample_prior, forward_model, compute_distance1)

@time out2 = abc_standard(summaryStatistics2, N, threshold2, sample_prior, forward_model, compute_distance2)

R"plot(density($(out1.particles)))"
R"lines(density($(out2.particles[:, 1])), lty = 2)"

R"hist($(out1.distances))"
R"hist($(out2.distances[:, 1]))"
R"hist($(out2.distances[:, 2]))"

out1.threshold
out1.testedSamples

out2.threshold
out2.testedSamples
