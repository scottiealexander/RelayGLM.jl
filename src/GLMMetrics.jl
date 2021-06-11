module GLMMetrics

using Statistics, LinearAlgebra
import Statistics

using ..RelayUtils

export PerformanceMetric, ROCArea, JSDivergence, PRArea, BinomialLikelihood,
    RelativeBinomialLikelihood, RRI

export eval_and_store!, key_name, better_than, worst_value, findbest

abstract type PerformanceMetric end

@inline better_than(::Type{<:PerformanceMetric}, x::Real, y::Real) = x > y
@inline worst_value(::Type{<:PerformanceMetric}) = -Inf
# ============================================================================ #
function findbest(p::T) where T<:PerformanceMetric

    best = worst_value(T)
    kb = 1
    for k in eachindex(p.x)
        if better_than(T, p.x[k], best)
            best = p.x[k]
            kb = k
        end
    end
    return best, kb
end

# ============================================================================ #
struct ROCArea <: PerformanceMetric
    x::Vector{Float64}
end

ROCArea(nfold::Integer) = ROCArea(fill(NaN, nfold))

function eval_and_store!(r::ROCArea, y::AbstractVector{<:Real}, yp::AbstractVector{<:Real}, k::Integer)
    r.x[k], _ = RelayUtils.roca(y, yp)
    return r
end

Statistics.mean(r::ROCArea) = nanmean(r.x)
function Base.copy!(dst::ROCArea, src::ROCArea)
    dst.x .= src.x
    return dst
end
key_name(::Type{ROCArea}) = "roca"

# ============================================================================ #
struct PRArea <: PerformanceMetric
    x::Vector{Float64}
end

PRArea(nfold::Integer) = PRArea(fill(NaN, nfold))

function eval_and_store!(r::PRArea, y::AbstractVector{<:Real}, yp::AbstractVector{<:Real}, k::Integer)
    r.x[k], _ = RelayUtils.precision_recall(y, yp)
    return r
end

Statistics.mean(r::PRArea) = nanmean(r.x)
function Base.copy!(dst::PRArea, src::PRArea)
    dst.x .= src.x
    return dst
end
key_name(::Type{PRArea}) = "pra"

# ============================================================================ #
struct JSDivergence <: PerformanceMetric
    x::Vector{Float64}
    p::Vector{Float64}
end

JSDivergence(nfold::Integer) = JSDivergence(fill(NaN, nfold), fill(NaN, nfold))

function eval_and_store!(j::JSDivergence, y::AbstractVector{<:Real}, yp::AbstractVector{<:Real}, k::Integer)
    #                                                        niter, nbin
    j.x[k], j.p[k] = RelayUtils.permute_jensen_shannon(y, yp, 1000, 100)
    return j
end

Statistics.mean(j::JSDivergence) = nanmean(j.x), nanmean(j.p)
function Base.copy!(dst::JSDivergence, src::JSDivergence)
    dst.x .= src.x
    dst.p .= src.p
    return dst
end
key_name(::Type{JSDivergence}) = "jsd"
# ============================================================================ #
function binomial_lli(y::AbstractVector{<:Real}, yp::AbstractVector{<:Real})
    sm = 0.0
    @inbounds for k in eachindex(y)
        if y[k] > 0.0
            sm += log(yp[k] + eps())
        else
            sm += log(1.0 - (yp[k] - eps()))
        end
    end
    return sm
end
function binomial_lli(y::AbstractVector{<:Real})
    k = sum(y)
    n = length(y)

    # we assume mean efficacy is in (0,1)
    efficacy = k/n

    return k * log(efficacy) + (n-k) * log(1.0-efficacy)
end
# ============================================================================ #
struct BinomialLikelihood <: PerformanceMetric
    x::Vector{Float64}
end
BinomialLikelihood(nfold::Integer) = BinomialLikelihood(fill(NaN, nfold))

# NOTE: smaller is better for negative log likelihood of Binomial PDF
@inline better_than(::Type{BinomialLikelihood}, x::Real, y::Real) = x < y
@inline worst_value(::Type{BinomialLikelihood}) = +Inf

function eval_and_store!(r::BinomialLikelihood, y::AbstractVector{<:Real}, yp::AbstractVector{<:Real}, k::Integer)
    # r.x[k] = -(dot(log.(yp), y) + dot(1.0 .- y, log.(1.0 .- yp)))
    r.x[k] = -binomial_lli(y, yp) / length(y)
    return r
end

Statistics.mean(r::BinomialLikelihood) = nanmean(r.x)
function Base.copy!(dst::BinomialLikelihood, src::BinomialLikelihood)
    dst.x .= src.x
    return dst
end
key_name(::Type{BinomialLikelihood}) = "bili"
# ============================================================================ #
struct RelativeBinomialLikelihood <: PerformanceMetric
    x::Vector{Float64}
end
RelativeBinomialLikelihood(nfold::Integer) = RelativeBinomialLikelihood(fill(NaN, nfold))

# for relative likelihood, larger values are better
function eval_and_store!(r::RelativeBinomialLikelihood, y::AbstractVector{<:Real}, yp::AbstractVector{<:Real}, k::Integer)
    r.x[k] = (binomial_nlli(y, yp) - binomial_nlli(y)) / length(y) #null
    return r
end

Statistics.mean(r::RelativeBinomialLikelihood) = nanmean(r.x)
function Base.copy!(dst::RelativeBinomialLikelihood, src::RelativeBinomialLikelihood)
    dst.x .= src.x
    return dst
end
key_name(::Type{RelativeBinomialLikelihood}) = "rbili"
# ============================================================================ #
# Bernoulli variant of cross-validated (CV) single-spike information, which in
# this context becomes CV single-spike Bernoulli information *OR* the
# single-spike relative relay information (RRI)
# c.f. Williamson et al. (2015) PLOS Comp Bio, eq. 17, 20, and 59
struct RRI <: PerformanceMetric
    x::Vector{Float64}
end
RRI(nfold::Integer) = RRI(fill(NaN, nfold))

# for RRI, larger values are better
function eval_and_store!(r::RRI, y::AbstractVector{<:Real}, yp::AbstractVector{<:Real}, k::Integer)
    # divide by log(2) so our units are bits-per-event
    r.x[k] = (binomial_lli(y, yp) - binomial_lli(y)) / length(y) / log(2)
    return r
end

Statistics.mean(r::RRI) = nanmean(r.x)
function Base.copy!(dst::RRI, src::RRI)
    dst.x .= src.x
    return dst
end
key_name(::Type{RRI}) = "rri"
# ============================================================================ #
end
