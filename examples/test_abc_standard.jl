using ApproximateBayesianComputation, RCall

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

@time out00 = abc_standard(summaryStatistics1, N, 0.5, sample_prior, forward_model, compute_distance1)

@time out0 = abc_standard(summaryStatistics1, N, 0.25, sample_prior, forward_model, compute_distance1)

@time out1 = abc_standard(summaryStatistics1, N, 0.05, sample_prior, forward_model, compute_distance1)

@time out11 = abc_standard(summaryStatistics1, N, 0.01, sample_prior, forward_model, compute_distance1)

@time out12 = abc_standard(summaryStatistics1, N, 0.005, sample_prior, forward_model, compute_distance1)

@time out2 = abc_standard(summaryStatistics2, N, threshold2, sample_prior, forward_model, compute_distance2)

R"plot(density($(out12.particles)))"
R"lines(density($(out0.particles)))"
R"lines(density($(out00.particles)))"
R"lines(density($(out1.particles)))"
R"lines(density($(out11.particles)))"
R"abline(v = $summaryStatistics1)"

R"library(ggplot2)"
R"library(latex2exp)"

μ = sum(data) / (0.25) / (1.0 / 0.25 + 100.0 / 0.25)
σ = sqrt(1.0 / (1.0 / 0.25 + 100.0 / 0.25))

R"plot.df <- data.frame(eps5 = $(out00.particles),
eps25 = $(out0.particles),
eps05 = $(out1.particles),
eps01 = $(out11.particles),
eps005 = $(out12.particles))"
R"plot.df$x <- seq(1.0, 3.0, length.out = 200)"
R"plot.df$y <- dnorm(plot.df$x, $μ, $σ)"


R"ggplot(plot.df) +
 geom_density(aes(x = eps5)) +
 geom_line(aes(x = x, y = y), linetype = 2) +
 xlab(TeX('$\\theta$'))+
 annotate('text', x = 1.5, y = 3.45,
             label = TeX('$\\epsilon = 0.5$', output='character'),
 parse = TRUE) +
 ggtitle('Standard ABC Algorithm') +
 theme(plot.title = element_text(hjust = 0.5))"

R"ggplot(plot.df) +
 geom_density(aes(x = eps25)) +
 geom_line(aes(x = x, y = y), linetype = 2) +
 xlab(TeX('$\\theta$'))+
 annotate('text', x = 1.75, y = 4.0,
             label = TeX('$\\epsilon = 0.25$', output='character'),
 parse = TRUE) +
 ggtitle('Standard ABC Algorithm') +
 theme(plot.title = element_text(hjust = 0.5))"

R"ggplot(plot.df) +
 geom_density(aes(x = eps05)) +
 geom_line(aes(x = x, y = y), linetype = 2) +
 xlab(TeX('$\\theta$'))+
 annotate('text', x = 2.1, y = 6.0,
             label = TeX('$\\epsilon = 0.05$', output='character'),
 parse = TRUE) +
 ggtitle('Standard ABC Algorithm') +
 theme(plot.title = element_text(hjust = 0.5))"

R"ggplot(plot.df) +
 geom_density(aes(x = eps01)) +
 geom_line(aes(x = x, y = y), linetype = 2) +
 xlab(TeX('$\\theta$'))+
 annotate('text', x = 2.1, y = 6.0,
             label = TeX('$\\epsilon = 0.01$', output='character'),
 parse = TRUE) +
 ggtitle('Standard ABC Algorithm') +
 theme(plot.title = element_text(hjust = 0.5))"

R"ggplot(plot.df) +
 geom_density(aes(x = eps005)) +
 geom_line(aes(x = x, y = y), linetype = 2) +
 xlab(TeX('$\\theta$'))+
 annotate('text', x = 2.15, y = 6.0,
             label = TeX('$\\epsilon = 0.005$', output='character'),
 parse = TRUE) +
 ggtitle('Standard ABC Algorithm') +
 theme(plot.title = element_text(hjust = 0.5))"

R"lines(density($(out2.particles[:, 1])), lty = 2)"

R"hist($(out1.distances))"
R"hist($(out2.distances[:, 1]))"
R"hist($(out2.distances[:, 2]))"

out1.threshold
out1.testedSamples

out2.threshold
out2.testedSamples
