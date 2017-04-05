"""
Creates a sampler from distribution(s)
"""
function make_sample_prior{D <: Distributions.Distribution}(distr::D)
    return function sample_prior()
        return rand(distr)
    end
end

function make_sample_prior{D <: Distributions.Distribution}(distr::Array{D, 1})
    return function sample_prior()
        return rand.(distr)
    end
end

function make_sample_prior{T <: Tuple, D <: DataType}(param::T, distr::D)
    return function sample_prior()
        return rand(distr(param...))
    end
end

function make_sample_prior{T <: Tuple, D <: DataType}(param::Array{T, 1}, distr::D)
    d = [distr(p...) for p in param]
    return function sample_prior()
        return rand.(d)
    end
end

function make_sample_prior{T <: Tuple, D <: DataType}(param::Array{T, 1}, distr::Array{D, 1})
    dA = [d(p...) for (p, d) in zip(param, distr)]
    return function sample_prior()
        return rand.(dA)
    end
end

"""
Creates a density from distribution or set of distributions
"""
function make_density_prior{D <: Distributions.Distribution}(distr::D)
    return function density_prior(x::Float64)
        return Distributions.pdf(distr, x)
    end
end

function make_density_prior{D <: Distributions.Distribution}(distr::Array{D, 1})
    return function density_prior(x::Array{Float64, 1})
        tot = 0.0
        for (d, x) in zip(distr, x)
            tot += Distributions.logpdf(d, x)
        end 
        return exp(tot)
    end
end

"""
Simultaneously generate sampler and density function for distribution and parameters
"""
function make_model_prior{D <: Distributions.Distribution}(d::D)
    return make_sample_prior(d), make_density_prior(d)
end

function make_model_prior{D <: Distributions.Distribution}(d::Array{D, 1})
    return make_sample_prior(d), make_density_prior(d)
end

function make_model_prior{T <: Tuple, D <: DataType}(param::T, distr::D)
    d = distr(param...)
    return make_sample_prior(d), make_density_prior(d)
end

function make_model_prior{T <: Tuple, D <: DataType}(param::Array{T, 1}, distr::D)
    dA = [distr(p...) for p in param]
    return make_sample_prior(dA), make_density_prior(dA)
end

function make_model_prior{T <: Tuple, D <: DataType}(param::Array{T, 1}, d::Array{D, 1})
    dA = [d(p...) for (p, d) in zip(param, distr)]
    return make_sample_prior(dA), make_density_prior(dA)
end
