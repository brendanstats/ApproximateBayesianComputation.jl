
#Distributions.Normal()
#Distributions.TruncatedNormal()
#rand()
#Distributions.logpdf(d, x)
#Distributions.pdf(d, x)

#sample_kernel()
#density_kernel()

"""
Simultaneously generate sampler and density function for kernel
"""

function make_normal_kernel()
     function sample_kernel(x0::Float64, kernelbw::Float64)
        return randn() * kernelbw + x0
    end
     function density_kernel(x::Float64, x0::Float64, kernelbw::Float64)
        d = Distributions.Normal(x0, kernelbw)
        return Distributions.pdf(d, x)
     end
    return sample_kernel, density_kernel
end

function make_jointnormal_kernel()
     function sample_kernel(x0::Array{Float64, 1}, kernelbw::Array{Float64,1})
        return randn(length(x0)) .* kernelbw + x0
    end
    function density_kernel(x::Array{Float64, 1}, x0::Array{Float64, 1}, kernelbw::Array{Float64, 1})
        dA = [Distributions.Normal(x0i, ksd) for (x0i, ksd) in zip(x0, kernelbw)]
        logpA = [Distributions.logpdf(d, xi) for (d, xi) in zip(dA, x)]
        logp = sum(logpA)
        return exp(logp)
    end
    return sample_kernel, density_kernel
end

function make_truncatednormal_kernel(lowerbound::Float64, upperbound::Float64)
    function sample_kernel(x0::Float64, kernelbw::Float64)
        d = Distributions.TruncatedNormal(x0, kernelbw, lowerbound, upperbound)
        return rand(d)
    end
    function density_kernel(x::Float64, x0::Float64, kernelbw::Float64)
        d = Distributions.TruncatedNormal(x0, kernelbw, lowerbound, upperbound)
        return Distributions.pdf(d, x)
    end
    return sample_kernel, density_kernel
end


function make_truncatednormal_kernel(lowerbounds::Array{Float64, 1}, upperbounds::Array{Float64, 1})
    function sample_kernel(x0::Array{Float64, 1}, kernelbw::Array{Float64, 1})
        dA = [Distributions.TruncatedNormal(x0i, ksd, l, u) for (x0i, ksd, l, u) in zip(x0, kernelbw, lowerbounds, upperbounds)]
        return rand.(dA)
    end
    function density_kernel(x::Array{Float64, 1}, x0::Array{Float64, 1}, kernelbw::Array{Float64, 1})
        dA = [Distributions.TruncatedNormal(x0i, ksd, l, u) for (x0i, ksd, l, u) in zip(x0, kernelbw, lowerbounds, upperbounds)]
        logpA = [Distributions.logpdf(d, xi) for (d, xi) in zip(dA, x)]
        logp = sum(logpA)
        return exp(logp)
    end
    return sample_kernel, density_kernel
end

function make_truncatednormal_kernel(bounds::Array{Tuple{Float64, Float64}, 1})
    function sample_kernel(x0::Array{Float64, 1}, kernelbw::Array{Float64, 1})
        dA = [Distributions.TruncatedNormal(x0i, ksd, b...) for (x0i, ksd, b) in zip(x0, kernelbw, bounds)]
        return rand.(dA)
    end
    function density_kernel(x::Array{Float64, 1}, x0::Array{Float64, 1}, kernelbw::Array{Float64, 1})
        dA = [Distributions.TruncatedNormal(x0i, ksd, b...) for (x0i, ksd, b) in zip(x0, kernelbw, bounds)]
        logpA = [Distributions.logpdf(d, xi) for (d, xi) in zip(dA, x)]
        logp = sum(logpA)
        return exp(logp)
    end
    return sample_kernel, density_kernel
end


function make_joint_kernel(distr::Array{Any, 1}, params::Array{Tuple, 1})
    function sample_kernel(x0::Array{Float64, 1}, kernelbw::Array{Float64, 1})
        dA = [d(x0i, ksd, p...) for (d, x0i, ksd, p) in zip(distr, x0, kernelbw, params)]
        return rand.(dA)
    end
    function density_kernel(x::Array{Float64, 1}, x0::Array{Float64, 1}, kernelbw::Array{Float64, 1})
        dA = [d(x0i, ksd, p...) for (d, x0i, ksd, p) in zip(distr, x0, kernelbw, params)]
        logpA = [Distributions.logpdf(d, xi) for (d, xi) in zip(dA, x)]
        logp = sum(logpA)
        return exp(logp)
    end
    return sample_kernel, density_kernel
end

