module GLMFit

using SparseArrays, Random, LinearAlgebra, Optim

using ..RelayUtils, ..GLMMetrics

include("./GLMModels.jl")
using .GLMModels

include("./GLMPriors.jl")
using .GLMPriors

include("./Partitions.jl")
using .Partitions

export Poisson, Binomial, Gaussian, Identity, Exponential, Logistic
export Distribution
export ActivationFunction

export coefficients, metric, nlli, converged, lambda

mutable struct CrossValResult{T<:PerformanceMetric}
    coef::Vector{Float64}
    metric::T
    li::Float64
    nconvg::Int
    ntotal::Int
    lambda::Vector{Float64}
end

function CrossValResult(metric::T, ncoef::Integer, nlambda::Integer) where T <: PerformanceMetric
    return CrossValResult{T}(zeros(ncoef), metric, +Inf, 0, 0, zeros(nlambda))
end

@inline coefficients(x::CrossValResult) = x.coef
@inline metric(x::CrossValResult) = x.metric
@inline nlli(x::CrossValResult) = x.li
@inline converged(x::CrossValResult) = x.nconvg == x.ntotal
@inline lambda(x::CrossValResult) = x.lambda
# ============================================================================ #
function cross_validate(::Type{T}, d::RegularizedGLM{D,A}, pr::AbstractPrior, x0::Vector{<:Real},
    nfold::Integer, shlf::Bool=false) where {D,A,T<:PerformanceMetric}

    result = CrossValResult(T(nfold), nvar(d), nlambda(pr))

    kmin = 1
    # pm = allocate_prior(pr)

    for k in 1:length(pr)

        get_prior!(d.prior, pr, k)

        res = cross_validate(T, d, x0, nfold, shlf)

        if nlli(res) < nlli(result)
            result.coef .= res.coef
            copy!(result.metric, res.metric)
            result.li = res.li
            result.nconvg = res.nconvg
            result.ntotal = res.ntotal
            result.lambda .= get_lambda(pr, k)
            kmin = k
        end
    end

    # if kmin == 1
    #     @warn("Smallest lambda selected!")
    # end

    return result
end
# ============================================================================ #
# function cross_validate(::Type{T}, d::GLM{D,A}, pr0::SparseMatrixCSC{Float64,Int}, x0::Vector{<:Real},
#     nfold::Integer, shfl::Bool=false) where {D,A,T<:PerformanceMetric}
function cross_validate(::Type{T}, d::GLMType, x0::Vector{<:Real},
    nfold::Integer, shfl::Bool=false) where {T<:PerformanceMetric,GLMType<:AbstractGLM,D,A}

    coef = zeros(length(x0))
    li = 0.0
    ntotal = 0
    nconvg = 0

    res = T(nfold)

    # # TODO we should shift this conditional to dispatch
    # use_prior = !iszero(pr0)

    for p in ballanced_partition(Partitioner, predictors(d), response(d), nfold, shfl)
        convg = 1
        xtrain, ytrain = training_set(p)
        xtest, ytest = testing_set(p)

        glm = init_glm(d, xtrain, ytrain, bin_size(d))

        # the correct fit() will be compiled based on the type of GLM object
        # (i.e. GLMType)
        xmin, convg = fit(glm, x0)

        # we want a un-regularized GLM for assessing cross-validated
        # likelihood reguardless of regularization during training
        glm_test = GLM(d, xtest, ytest, bin_size(d))

        # syntax is funky to just get the likelihood due to Optim weirdness
        # with Optim.only_fg!() / Optim.only_fgh!() syntax, 0.0 just means
        # compute objective (nothing -> don't compute objective), 4th input
        # indicates whether or not to compute gradient (nothing -> don't) 3rd
        # input is parameter vector
        li += GLMModels.nlli!(glm_test, 0.0, xmin, nothing)

        pred, _ = GLMModels.predict(glm_test, xmin)

        if all(isfinite, pred)
            coef .+= xmin
            ntotal += 1

            eval_and_store!(res, ytest, pred, ntotal)

            if convg > 0
                nconvg += 1
            end
        end
    end

    # NOTE WARNING TODO
    # does it make the most sense to average the coefficients? or the kernels?
    # probably doesn't matter but that should be confirmed
    coef ./= ntotal
    li /= ntotal

    return CrossValResult{T}(coef, res, li, nconvg, ntotal, [0.0])
end
# ============================================================================ #
@inline fit(d::GLM{Gaussian,Identity}, p0::Vector{<:Real}) = vec(pinv(d.x'*d.x) * (d.x' * d.y)), 1#d.x' * d.y ./ sum(d.y)
# ---------------------------------------------------------------------------- #
@inline function fit(d::AbstractGLM, p0::Vector{<:Real})
    xmin, convg = fit(d, p0, 2_000)
    if convg < 1
        xmin, convg = fit(d, xmin, 4_000)
    end
    return xmin, convg
end
# ---------------------------------------------------------------------------- #
function fit(d::AbstractGLM, p0::Vector{<:Real}, it::Integer)

    converged = 1
    res = optimize(Optim.only_fg!(get_objective(d)), p0, LBFGS(),
        Optim.Options(allow_f_increases=true, iterations=it, f_abstol=1e-8)
    )

    if !Optim.converged(res) && !convergence_check(res)
        # Optim.converged(x) -> Optim.x_converged(x) || Optim.f_converged(x) || Optim.g_converged(x)
        # show(res)
        converged = 0
    end

    return Optim.minimizer(res), converged
end
# ---------------------------------------------------------------------------- #
function fit(d::T, p0::Vector{<:Real}, it::Integer) where T<:AbstractGLM{Binomial,Logistic}

    converged = 1
    res = optimize(Optim.only_fgh!(get_objective(d)), p0, Newton(),
        Optim.Options(allow_f_increases=true, iterations=it, f_abstol=1e-8)
    )

    if !Optim.converged(res) && !convergence_check(res)
        # Optim.converged(x) -> Optim.x_converged(x) || Optim.f_converged(x) || Optim.g_converged(x)
        # show(res)
        converged = 0
    end

    return Optim.minimizer(res), converged
end
# ============================================================================ #
function convergence_check(res::Optim.MultivariateOptimizationResults)#{<:LBFGS})
# Optim.converged seems to return false sometimes even when the condition for
# f_abstol appears to have been met, so do the check manually

    return (res.x_abschange <= res.x_abstol) ||
           (res.x_relchange <= res.x_reltol) ||
           (res.f_abschange <= res.f_abstol) ||
           (res.f_relchange <= res.f_reltol) ||
           (res.g_residual  <= res.g_abstol)
end
# ============================================================================ #
end  # module GLMFit
