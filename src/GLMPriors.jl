module GLMPriors

using SparseArrays, LinearAlgebra

export ridge_prior, smoothing_prior, null_prior, get_prior!, get_lambda, nlambda
export allocate_prior, AbstractPrior

const SPMat = SparseMatrixCSC{Float64,Int}
# ============================================================================ #
abstract type AbstractPrior{N,T} end
struct GLMPrior{N,T} <: AbstractPrior{N,T}
    d::SPMat
    filter_len::NTuple{N,Int}
    lambda::NTuple{T,Vector{Float64}}
    lambda_len::NTuple{T,Int}
end
function GLMPrior{N,T}(d::SPMat, x::NTuple{N,<:Integer},
    lm::NTuple{T,AbstractVector{<:Real}}) where {N,T}
    if ((T < N) && T != 1)
        error("Invalid number of lambda parameters")
    elseif T >= N
        T2 = N
    else
        T2 = T
    end

    return GLMPrior{N,T2}(d, x, lm[1:T2], map(length, lm[1:T2]))
end
# ============================================================================ #
struct NullPrior{N} <: AbstractPrior{N, 0}
    filter_len::NTuple{N,Int}
end
# ============================================================================ #
struct LassoPrior{N,T} <: AbstractPrior{N,T}
    filter_len::NTuple{N,Int}
    lambda::NTuple{T,Vector{Float64}}
    lambda_len::NTuple{T,Int}
end
struct Lasso{T<:Real}
    x::Vector{T}
end
Base.:*(v::AbstractArray{<:Number}, l::Lasso) = sign.(v) .* reshape(l.x, size(v))
Base.:*(l::Lasso, v::AbstractArray{<:Number}) = sign.(v) .* reshape(l.x, size(v))
# ============================================================================ #
@inline nlambda(gp::GLMPrior{N,T}) where {N,T} = T
@inline nlambda(gp::NullPrior) = 0
@inline nlambda(gp::LassoPrior{N,T}) where {N,T} = T
# ---------------------------------------------------------------------------- #
@inline Base.length(gp::AbstractPrior) = prod(gp.lambda_len)
@inline Base.length(gp::NullPrior) = 1
# ---------------------------------------------------------------------------- #
function get_prior!(x::SPMat, gp::NullPrior, k::Integer)
    knz = findall(!iszero, x)
    x[knz] .= 0.0
    return x
end
# ---------------------------------------------------------------------------- #
function get_prior!(x::SPMat, gp::GLMPrior{N,T}, k::Integer) where {N,T}
    kstart = 0
    kend = 1

    kp = CartesianIndices(gp.lambda_len)[k]
    inc = 1
    for kl in (T < N ? fill(1, N) : 1:N)
        kstart = kend + 1
        kend = kstart + gp.filter_len[inc] - 1
        slice = kstart:kend
        x[:, slice] .= gp.d[:, slice] .* gp.lambda[kl][kp[kl]]
        inc += 1
    end
    return x
end
# ---------------------------------------------------------------------------- #
function get_prior!(l::Lasso, gp::LassoPrior{N,T}, k::Integer) where {N,T}
    kstart = 0
    kend = 1

    kp = CartesianIndices(gp.lambda_len)[k]
    inc = 1
    for kl in (T < N ? fill(1, N) : 1:N)
        kstart = kend + 1
        kend = kstart + gp.filter_len[inc] - 1
        slice = kstart:kend
        l.x[slice] .= gp.lambda[kl][kp[kl]]
        inc += 1
    end
    return l
end
# ---------------------------------------------------------------------------- #
function allocate_prior(gp::AbstractPrior)
    n = sum(gp.filter_len)+1
    return spzeros(n, n)
end
# ---------------------------------------------------------------------------- #
function allocate_prior(gp::LassoPrior)
    n = sum(gp.filter_len)+1
    return Lasso(zeros(n))
end
# ---------------------------------------------------------------------------- #
function get_lambda(gp::AbstractPrior, k::Integer)
    kp = CartesianIndices(gp.lambda_len)[k]
    out = Vector{Float64}(undef, length(gp.lambda))
    for k in eachindex(out)
        out[k] = gp.lambda[k][kp[k]]
    end
    return out
end
get_lambda(gp::NullPrior, k::Integer) = [0.0]
# ============================================================================ #
@inline null_prior(x::NTuple{N,<:Integer}) where N = NullPrior(x)
# ---------------------------------------------------------------------------- #
function ridge_prior(x::NTuple{N,<:Integer},
    lm::NTuple{T,AbstractVector{<:Real}}) where {N,T}
    n = sum(x)
    return GLMPrior{N,T}(blockdiag(spzeros(1,1), sparse(I, n, n)), x, lm)
end
# ---------------------------------------------------------------------------- #
function smoothing_prior(x::NTuple{N,<:Integer},
    lm::NTuple{T,AbstractVector{<:Real}}) where {N,T}

    tmp = Vector{SparseMatrixCSC}(undef, length(x))
    for k in eachindex(x)
        d = spdiagm(0 => fill(-1.0, x[k]), 1 => fill(1.0, x[k]-1))
        tmp[k] = d'*d
        tmp[k][end,end] = 1.0
    end
    bd = blockdiag(spzeros(1,1), tmp...)
    return GLMPrior{N,T}(blockdiag(spzeros(1,1), tmp...), x, lm)
end
# ---------------------------------------------------------------------------- #
function lasso_prior(x::NTuple{N,<:Integer},
    lm::NTuple{T,AbstractVector{<:Real}}) where {N,T}

    return LassoPrior(x, map(collect, lm), map(length, lm))
end
# ============================================================================ #
end
