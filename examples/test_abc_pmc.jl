using ApproximateBayesianComputation

#Start Testing...
data = rand(Distributions.Normal(2.0, 0.5), 100)

summaryStatistics = mean(data)
nsteps = 15
numParticles = 200
initialSample = 1000

sample_prior, density_prior = make_model_prior(Distributions.Normal(0.0, 0.5))
sample_kernel, density_kernel = make_normal_kernel()
kernel_sd = weightedstd2

compute_distance(x::Array{Float64, 1}, y::Float64) = abs(mean(x) - y)
forward_model(μ::Float64) = rand(Distributions.Normal(μ, 0.5), 100)

rank_distances = identity
shrink_threshold{A <: SingleMeasureAbcPmc}(abcpmc::A) = quantile_threshold(abcpmc, 0.3)

log = true
logFile = "log.txt"
save = true
saveFile = "results.jld"
verbose = true

out = abc_pmc(summaryStatistics, nsteps, numParticles, initialSample, sample_prior,
              density_prior, sample_kernel, density_kernel, compute_distance,
              forward_model, rank_distances, kernel_sd, shrink_threshold,
              verbose, log, logFile, save, saveFile)

using RCall

R"library(ggplot2)"

R"step1 <- data.frame(step = '1', particles = $(out[1].particles), weights = $(out[1].weights.values), distances = $(out[1].distances))"
R"step2 <- data.frame(step = '2', particles = $(out[2].particles), weights = $(out[2].weights.values), distances = $(out[2].distances))"
R"step5 <- data.frame(step = '5', particles = $(out[5].particles), weights = $(out[5].weights.values), distances = $(out[1].distances))"
R"step8 <- data.frame(step = '8', particles = $(out[8].particles), weights = $(out[8].weights.values), distances = $(out[8].distances))"
R"step12 <- data.frame(step = '12', particles = $(out[12].particles), weights = $(out[12].weights.values), distances = $(out[12].distances))"
R"step15 <- data.frame(step = '15', particles = $(out[15].particles), weights = $(out[15].weights.values), distances = $(out[15].distances))"
R"pmc.df <- rbind(step1, step2, step5, step8, step12, step15)"


R"ggplot(pmc.df, aes(x = particles, weight = weights, group = step, color = step)) +
 geom_density()"

R"ggplot(pmc.df, aes(x = particles, y = log(distances), color = weights)) +
 geom_point() + geom_vline(xintercept = mean($data), alpha = 0.5, linetype = 2) +
 facet_wrap(~step)"

x = collect(range(-1.0, 0.01, 351))
pd = density_prior.(x)

steps = [1, 2, 5, 8, 12, 15]
kd = zeros(Float64, length(x), length(steps))
kda = zeros(Float64, length(x), length(steps))
kdo = zeros(Float64, length(x), length(steps))
for (ii, ss) in enumerate(steps)
    kd[:, ii] = compute_density(x, out[ss], kernel_sd, density_kernel)
    wadj = distance_weights(out[ss])
    kda[:, ii] = compute_density(x, out[ss], wadj, kernel_sd, density_kernel)
    if ss > 1
        wopt = densitysq_weights(out[ss], out[ss - 1], kernel_sd, density_kernel)
        kdo[:, ii] = compute_density(x, out[ss], wopt, kernel_sd, density_kernel)
    end
end

R"kernel.df <- rbind(data.frame(step = 'prior', x = $x, y = $pd),
 data.frame(step = paste('step', $steps[1]), x = $x, y = $kd[, 1]),
 data.frame(step = paste('step', $steps[2]), x = $x, y = $kd[, 2]),
 data.frame(step = paste('step', $steps[3]), x = $x, y = $kd[, 3]),
 data.frame(step = paste('step', $steps[4]), x = $x, y = $kd[, 4]),
 data.frame(step = paste('step', $steps[5]), x = $x, y = $kd[, 5]),
 data.frame(step = paste('step', $steps[6]), x = $x, y = $kd[, 6]))"
R"kernel.df$type = 'standard'"

R"kerneladj.df <- rbind(data.frame(step = 'prior', x = $x, y = $pd),
 data.frame(step = paste('step', $steps[1]), x = $x, y = $kda[, 1]),
 data.frame(step = paste('step', $steps[2]), x = $x, y = $kda[, 2]),
 data.frame(step = paste('step', $steps[3]), x = $x, y = $kda[, 3]),
 data.frame(step = paste('step', $steps[4]), x = $x, y = $kda[, 4]),
 data.frame(step = paste('step', $steps[5]), x = $x, y = $kda[, 5]),
 data.frame(step = paste('step', $steps[6]), x = $x, y = $kda[, 6]))"
R"kerneladj.df$type = 'distance-weighted'"

R"kernelopt.df <- rbind(
 data.frame(step = paste('step', $steps[2]), x = $x, y = $kdo[, 2]),
 data.frame(step = paste('step', $steps[3]), x = $x, y = $kdo[, 3]),
 data.frame(step = paste('step', $steps[4]), x = $x, y = $kdo[, 4]),
 data.frame(step = paste('step', $steps[5]), x = $x, y = $kdo[, 5]),
 data.frame(step = paste('step', $steps[6]), x = $x, y = $kdo[, 6]))"
R"kernelopt.df$type = 'accelerated'"

R"ggplot(kernel.df, aes(x = x, y = y, color = step)) + geom_line()"

R"ggplot(rbind(kernel.df, kerneladj.df),
 aes(x = x, y = y, color = step, linetype = type)) + geom_line()"

R"ggplot(rbind(kernel.df, kernelopt.df),
 aes(x = x, y = y, color = step, linetype = type)) + geom_line()"
