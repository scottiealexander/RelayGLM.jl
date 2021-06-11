include("./basis_types.jl")
include("./GLMFit.jl")
using .GLMFit
using .GLMFit.GLMMetrics
import .GLMFit.GLMPriors
import .RelayUtils
using LinearAlgebra, Statistics, Random
const VecVec{T} = AbstractVector{<:AbstractVector{T}}
# ============================================================================ #
struct Predictor{T<:AbstractBasis}
    ts::Vector{Float64}
    evt::Vector{Float64}
    basis::T
end

Predictor(x::Vector{Float64}) = Predictor(Float64[], x, NullBasis())

function Predictor(p::Predictor{L}, basis::T) where {L<:AbstractBasis, T<:AbstractBasis}
    T == L && return p
    return Predictor(p.ts, p.evt, basis)
end

@inline nrow(p::Predictor) = length(p.evt)
@inline ncol(p::Predictor) = nbasis(p.basis)
@inline Base.size(p::Predictor) = (nrow(p), ncol(p))

@inline generate(p::Predictor) = project(p.basis, p.ts, p.evt)

@inline get_basis(p::Predictor) = p.basis

# ============================================================================ #
struct PredictorSet
    keys::Vector{Symbol}
    d::Dict{Symbol, Predictor}
end

PredictorSet() = PredictorSet(Vector{Symbol}(), Dict{Symbol, Predictor}())

Base.IteratorSize(::PredictorSet) = Base.HasLength()
Base.IteratorEltype(::PredictorSet) = Base.HasEltype()
Base.eltype(p::PredictorSet) = Predictor
Base.length(p::PredictorSet) = length(p.keys)
Base.iterate(p::PredictorSet) = iterate(p, 1)
function Base.iterate(p::PredictorSet, k::Integer)
    k > length(p) && return nothing
    return p.d[p.keys[k]], k+1
end

Base.getindex(ps::PredictorSet, k::Symbol) = ps.d[k]
function Base.setindex!(ps::PredictorSet, p::Predictor, k::Symbol)
    !in(k, ps.keys) && push!(ps.keys, k)
    ps.d[k] = p
    return ps
end

@inline ncol(ps::PredictorSet) = sum(ncol, ps) + 1

function generate(ps::PredictorSet)
    nc = ncol(ps)
    nr = nrow(first(ps))
    @assert(all(x -> nrow(x) == nr, ps), "Predictors do not have a common number of observations (rows)!")

    # NOTE: the y-intercept is addeded here as the first column
    dm = ones(nr, nc)
    kl = 2
    for p in ps
        ke = kl + ncol(p) - 1
        dm[:,kl:ke] .= generate(p)
        kl = ke + 1
    end
    return dm
end

@inline get_basis(p::PredictorSet, k::Symbol) = get_basis(p[k])
# ============================================================================ #
abstract type AbstractPrior end
struct NullPrior <: AbstractPrior end
struct RidgePrior <: AbstractPrior end
struct SmoothingPrior <: AbstractPrior end
struct LassoPrior <: AbstractPrior end

generate(::Type{NullPrior}, flen::NTuple{N,Int}, lmb::NTuple{N,T}) where {N,T<:AbstractVector{<:Real}} = GLMPriors.null_prior(flen)
generate(::Type{RidgePrior}, flen::NTuple{N,Int}, lmb::NTuple{N,T}) where {N,T<:AbstractVector{<:Real}} = GLMPriors.ridge_prior(flen, lmb)
generate(::Type{SmoothingPrior}, flen::NTuple{N,Int}, lmb::NTuple{N,T}) where {N,T<:AbstractVector{<:Real}} = GLMPriors.smoothing_prior(flen, lmb)
generate(::Type{LassoPrior}, flen::NTuple{N,Int}, lmb::NTuple{N,T}) where {N,T<:AbstractVector{<:Real}} = GLMPriors.lasso_prior(flen, lmb)

# ============================================================================ #
abstract type AbstractGLM end

struct GLM <: AbstractGLM #{T<:AbstractPrior,R<:Real,L<:Real}
    predictors::PredictorSet
    response::Vector{Float64}
    # lambda::Vector{Vector{L}}
end
struct RegGLM{T<:AbstractPrior} <: AbstractGLM
    glm::GLM
    lambda::Vector{Vector{Float64}}
end
"""
lambda = [1:10,[]] # -> apply prior to only the first filter, ignore 2nd
lambda = [1:10] # -> apply lambda to all filters
"""
# function GLM(ps::PredictorSet, y::Vector{<:Real})
#     return GLM(ps, y)
# end
function GLM(ps::PredictorSet, y::Vector{<:Real}, prior::Type{P}, lambda::VecVec{N}) where {P<:AbstractPrior,N<:Real}
    return RegGLM{P}(GLM(ps, y), map(collect, lambda))
end
function GLM(glm::GLM)
    return GLM(glm.predictors, glm.response)
end
function GLM(glm::GLM, prior::Type{P}, lambda::VecVec{N}) where {P<:AbstractPrior,N<:Real}
    return RegGLM{P}(glm, map(collect, lambda))
end
@inline get_basis(glm::GLM, name::Symbol) = get_basis(glm.predictors, name)
@inline get_predictor(glm::GLM, name::Symbol) = glm.predictors[name]
@inline get_keys(glm::GLM) = glm.predictors.keys
@inline predictors(glm::GLM) = glm.predictors
@inline response(glm::GLM) = glm.response

@inline get_basis(rg::RegGLM, name::Symbol) = get_basis(rg.glm.predictors, name)
@inline get_predictor(rg::RegGLM, name::Symbol) = rg.glm.predictors[name]
@inline get_keys(rg::RegGLM) = rg.glm.predictors.keys
@inline predictors(rg::RegGLM) = rg.glm.predictors
@inline response(rg::RegGLM) = rg.glm.response

@inline function generate_prior(rg::RegGLM{P}) where P
    # lambda = [1:10,[]] # -> apply prior to only the first filter, ignore 2nd
    # lambda = [1:10] # -> apply lambda to all filters
    # if lambda contains a single set apply it to all filters
    flen = length(rg.lambda) == 1 ? sum : map
    return generate(P, tuple(flen(ncol, rg.glm.predictors)...), tuple(rg.lambda...))
end
# ============================================================================ #
struct GLMFilter
    coef::Vector{Float64}
    error::Vector{Float64}
end
struct GLMResult{T<:PerformanceMetric}
    filters::Dict{Symbol,GLMFilter}
    lambda::Dict{Symbol,Float64}
    coef::Vector{Float64}
    metric::T
    nlli::Float64
    null_nlli::Float64
    converged::Bool
end
function GLMResult(coef::Vector{Float64}, metric::T, li::Real, nl::Real, convg::Bool) where T <: PerformanceMetric
    return GLMResult{T}(Dict{Symbol,GLMFilter}(), Dict{Symbol,Float64}(), coef, metric, li, nl, convg)
end

@inline function Base.setindex!(r::GLMResult, f::GLMFilter, k::Symbol)
    r.filters[k] = f
    return r
end
@inline get_coef(r::GLMResult, k::Symbol) = r.filters[k].coef
@inline get_error(r::GLMResult, k::Symbol) = r.filters[k].error
@inline metric(r::GLMResult) = r.metric
@inline function set_lambda!(r::GLMResult, k::Symbol, v::Float64)
    r.lambda[k] = v
    return r
end
@inline get_lambda(r::GLMResult, k::Symbol) = r.lambda[k]
# ============================================================================ #
wasrelayed(pre::Vector{Float64}, post::Vector{Float64}) = RelayUtils.relay_status(pre, post)
# ============================================================================ #
function cross_validate_impl(::Type{T}, ::Type{D}, ::Type{A}, glm::GLM, nfold::Integer, shfl::Bool) where {T<:PerformanceMetric,D<:Distribution,A<:ActivationFunction}
    bs = map(x -> getfield(get_basis(x), :bin_size), predictors(glm))
    @assert(all(x -> x==bs[1], bs), "Bin sizes *MUST* be consistent across predictors")

    # construct GLM data container from the old interface
    data = GLMFit.GLM(D, A, generate(predictors(glm)), glm.response, bs[1])

    # initial guess for model parameters is just the "relayed-spike triggered average"
    x0 = data.x' * data.y ./ sum(data.y)

    # old cross_validate() interface does the actual fitting
    return GLMFit.cross_validate(T, data, x0, nfold, shfl), data
end
#----------------------------------------------------------------------------- #
function cross_validate_impl(::Type{T}, ::Type{D}, ::Type{A}, glm::RegGLM, nfold::Integer, shfl::Bool) where {T<:PerformanceMetric,D<:Distribution,A<:ActivationFunction}
    bs = map(x -> getfield(get_basis(x), :bin_size), predictors(glm))
    @assert(all(x -> x==bs[1], bs), "Bin sizes *MUST* be consistent across predictors")

    prior = generate_prior(glm)
    pr = GLMPriors.allocate_prior(prior)

    # construct GLM data container from the old interface
    data = GLMFit.RegularizedGLM(D, A, generate(predictors(glm)), glm.glm.response, bs[1], pr)

    # initial guess for model parameters is just the "relayed-spike triggered average"
    x0 = data.glm.x' * data.glm.y ./ sum(data.glm.y)

    # old cross_validate() interface does the actual fitting
    return GLMFit.cross_validate(T, data, prior, x0, nfold, shfl), data.glm
end
#----------------------------------------------------------------------------- #
function cross_validate(::Type{T}, ::Type{D}, ::Type{A}, glm::GLMType; nfold=10, shuffle_design=true) where {T<:PerformanceMetric,D<:Distribution,A<:ActivationFunction,GLMType<:AbstractGLM}

    res, data = cross_validate_impl(T, D, A, glm, nfold, shuffle_design)

    coef = coefficients(res)

    se = try
        # standard error of the optimization
        sqrt.(diag(pinv(GLMFit.hessian(data, coef))))
    catch err
        @warn("Failed to calculate coefficient std-error: \"$(typeof(err))\"")
        fill(NaN, length(coef))
    end

    # cor(coef, x0) > 0.9 && @warn("Fitting may have failed: cor(coef, x0) = $(cor(coef, x0))")

    result = GLMResult(
        coef,
        GLMFit.metric(res),
        GLMFit.nlli(res),
        GLMModels.null_nlli(data) / length(data.y),
        res.nconvg == res.ntotal
        )

    klambda = 1
    kl = 2
    lm = lambda(res)
    for k in get_keys(glm)
        basis = get_basis(glm, k)
        ke = kl + nbasis(basis) - 1
        result[k] = GLMFilter(basis * coef[kl:ke], basis * se[kl:ke])
        if klambda > length(lm)
            set_lambda!(result, k, lm[end])
        else
            set_lambda!(result, k, lm[klambda])
        end
        kl = ke + 1
        klambda += 1
    end

    return result
end
# ============================================================================ #
function jensen_shannon(::Type{A}, glm::GLM, result::GLMResult) where A <: ActivationFunction
    bs = get_basis(glm.predictors[:ff]).bin_size
    x = generate(glm.predictors)

    l = x * result.coef

    pred, _ = GLMModels.activate(A, l, bs)

    krel = findall(isequal(1), glm.response)
    kfail = findall(iszero, glm.response)

    d = RelayUtils.jensen_shannon(pred, glm.response)
    d2 = RelayUtils.jensen_shannon(l, glm.response)

    return d, d2
end
# ============================================================================ #
