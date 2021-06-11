module GLMModels

using SparseArrays, LinearAlgebra, Statistics

export Poisson, Binomial, Gaussian, Identity, Exponential, Logistic,
    nobs, nvar, activate!, get_objective, hessian, init_glm, predictors, response,
    bin_size

export Distribution, ActivationFunction, AbstractGLM, GLM, RegularizedGLM

const SPMat = SparseMatrixCSC{Float64,Int64}
const StorageVec = Union{Vector{Float64},Nothing}
const StorageMat = Union{Matrix{Float64},Nothing}
const ObjFlag = Union{Float64,Nothing}

abstract type Distribution end
struct Poisson <: Distribution end
struct Binomial <: Distribution end
struct Gaussian <: Distribution end

abstract type ActivationFunction end
struct Exponential <: ActivationFunction end
struct Logistic <: ActivationFunction end
struct Identity <: ActivationFunction end
# ============================================================================ #
abstract type AbstractGLM{D<:Distribution, A<:ActivationFunction} end
# ---------------------------------------------------------------------------- #
struct GLM{D,A} <: AbstractGLM{D,A}
    x::Matrix{Float64}
    y::Vector{Float64}
    r::Vector{Float64}
    dr::Vector{Float64}
    bin_size::Float64
end
# ---------------------------------------------------------------------------- #
struct RegularizedGLM{D,A,P<:AbstractMatrix} <: AbstractGLM{D,A}
    glm::GLM{D,A}
    prior::P
end
# ---------------------------------------------------------------------------- #
# main GLM constructor
function GLM(::Type{D}, ::Type{A}, x::Matrix{<:Real}, y::Vector{Float64}, bin_size::Real) where {D,A}
    return GLM{D,A}(x, y, zeros(Float64, length(y)), zeros(Float64, length(y)), bin_size)
end
# ---------------------------------------------------------------------------- #
# GLM copy constructor for easy partitioning
function GLM(glm::GLM{D,A}, x::Matrix{<:Real}, y::Vector{Float64}, bin_size::Real) where {D,A}
    return GLM(D, A, x, y, bin_size)
end
# ---------------------------------------------------------------------------- #
# construct a GLM from a RegularizedGLM, for use when training utilizes
# regularization but testing does not (which is always the case)
function GLM(rg::RegularizedGLM, x::Matrix{<:Real}, y::Vector{Float64}, bin_size::Real)
    return GLM(rg.glm, x, y, bin_size)
end
# ---------------------------------------------------------------------------- #
function RegularizedGLM(::Type{D}, ::Type{A}, x::Matrix{<:Real}, y::Vector{Float64}, bin_size::Real, pr::P) where {D,A,P}
    return RegularizedGLM{D,A,P}(GLM(D, A, x, y, bin_size), pr)
end
# ---------------------------------------------------------------------------- #
# alternate copy constructors for convienence use in GLMFit.cross_validate
function init_glm(glm::GLM{D,A}, x::Matrix{<:Real}, y::Vector{Float64}, bin_size::Real) where {D,A}
    return GLM(D, A, x, y, bin_size)
end
# ---------------------------------------------------------------------------- #
function init_glm(glm::RegularizedGLM{D,A}, x::Matrix{<:Real}, y::Vector{Float64}, bin_size::Real) where {D,A}
    return RegularizedGLM(D, A, x, y, bin_size, glm.prior)
end
# ============================================================================ #
@inline nobs(d::GLM) = length(d.y)
@inline nvar(d::GLM) = size(d.x, 2)
@inline nobs(d::RegularizedGLM) = length(d.glm.y)
@inline nvar(d::RegularizedGLM) = size(d.glm.x, 2)
@inline predictors(d::GLM) = d.x
@inline response(d::GLM) = d.y
@inline bin_size(d::GLM) = d.bin_size
@inline predictors(d::RegularizedGLM) = d.glm.x
@inline response(d::RegularizedGLM) = d.glm.y
@inline bin_size(d::RegularizedGLM) = d.glm.bin_size
# ============================================================================ #
get_objective(d::AbstractGLM) = (f::ObjFlag, g::StorageVec, p::Vector{Float64}) -> nlli!(d, f, p, g)

# function get_objective(d::GLM, pr::AbstractMatrix)
#     return (f::ObjFlag, g::StorageVec, p::Vector{Float64}) -> nlli_prior!(d, f, p, g, pr)
# end

get_objective(d::AbstractGLM{Binomial,Logistic}) =
    (f::ObjFlag, g::StorageVec, h::StorageMat, p::Vector{Float64}) -> nlli!(d, f, p, g, h)

# get_objective(d::GLM{Binomial,Logistic}, pr::AbstractMatrix) =
#     (f::ObjFlag, g::StorageVec, h::StorageMat, p::Vector{Float64}) -> nlli_prior!(d, f, p, g, h, pr)
# ============================================================================ #
function nlli!(d::RegularizedGLM{D,A,SPMat}, f::ObjFlag, p::Vector{Float64}, g::Vector{Float64}) where {D,A}
    nli = nlli!(d.glm, f, p, g)

    # is Optim requesting a gradient?
    if g != nothing
        g .+= d.prior * p
    end

    # Optim requested gradient only, don't bother calculating full objective
    f == nothing && return +Inf

    nli += .5 * (p' * d.prior * p)
    return nli
end
# ============================================================================ #
function nlli!(d::RegularizedGLM{D,A,SPMat}, f::ObjFlag, p::Vector{Float64}, g::StorageVec, h::StorageMat) where {D,A}
    nli = nlli!(d.glm, f, p, g, h)

    # is Optim requesting a gradient?
    if g != nothing
        g .+= d.prior * p
    end

    if h != nothing
        h .+= d.prior
    end

    # Optim requested gradient only, don't bother calculating full objective
    f == nothing && return +Inf

    nli += .5 * (p' * d.prior * p)
    return nli
end
# ============================================================================ #
function nlli!(d::RegularizedGLM{D,A,Matrix{Float64}}, f::ObjFlag, p::Vector{Float64}, g::Vector{Float64}) where {D,A}
    nli = nlli!(d.glm, f, p, g)

    # is Optim requesting a gradient?
    if g != nothing
        # g .+= cin * p
        LinearAlgebra.BLAS.gemv!('N', 1.0, d.prior, p, 1.0, g)
    end

    # Optim requested gradient only, don't bother calculating full objective
    f == nothing && return +Inf

    nli += .5 * (p' * d.prior * p)
    return nli
end
# ============================================================================ #
@inline function predict(d::RegularizedGLM, p::Vector{Float64})
    return predict(d.glm, p)
end
# ---------------------------------------------------------------------------- #
@inline function predict(d::GLM{D,L}, p::Vector{Float64}) where {D<:Distribution,L<:ActivationFunction}
    mul!(d.r, d.x, p)
    return activate!(L, d.r, d.dr, d.bin_size)
end
# ---------------------------------------------------------------------------- #
@inline function predict(d::GLM{D,Identity}, p::Vector) where D<:Distribution
    mul!(d.r, d.x, p)
    return d.r, d.dr
end
# ============================================================================ #
function nlli!(d::GLM{Gaussian,L}, p::Vector, g::StorageVec) where L <: ActivationFunction
    error("Not implemented!")
end
# ============================================================================ #
# source: https://www.comp.nus.edu.sg/~cs5240/lecture/matrix-diff.pdf slide 32 - 33
function nlli!(d::GLM{Gaussian,Identity}, f::ObjFlag, p::Vector{Float64}, g::StorageVec)
    r, _ = predict(d, p)

    # is Optim requesting a gradient?
    if g != nothing
        # Gaussian-Identity maximum likelyhood gradient: 2X'Xp - 2X'y
        # our approach is to rearrange the gradient as:
        # 2X'Xp - 2X'y = 2X'(Xp - y)

        # compute the term in the parenthese
        # d.r = Xp - y
        d.r .= d.y
        LinearAlgebra.BLAS.gemv!('N', 1.0, d.x, p, -1.0, d.r)

        # update the output variable <g> as
        # g = 2X'd.r
        LinearAlgebra.BLAS.gemv!('T', 2.0, d.x, d.r, 0.0, g)
    end

    # Optim requested gradient only, don't bother calculating full objective
    f == nothing && return +Inf

    # -log(exp(-(y .- r).^2))
    return sum(abs2, d.y .- r)
end
# ============================================================================ #
null_nlli(d::GLM{Gaussian,Identity}) = sum(abs2, d.y .- mean(d.y))
# ============================================================================ #
@inline function dnlli!(d::GLM{Poisson,Exponential}, r::Vector{Float64},
    dr::Vector{Float64}, g::Vector{Float64})

    # b/c exponential is it's own derivitive, r is likely to be aliased with
    # dr, thus we can simplify the gradient calculation that is used for
    # Poisson with any other activation function (see next impl. of dnlli!() below)
    # (i.e. dr / r == 1)

    # less efficent procedure that we implement below
    # g .= (d.x' * r) .- (d.x' * d.y)

    # second term g .= (d.x' * d.y)
    mul!(g, d.x', d.y)

    # first term and update (g .= d.x' * r -. g)
    LinearAlgebra.BLAS.gemv!('T', 1.0, d.x, r, -1.0, g)

    return g
end
# ============================================================================ #
@inline function dnlli!(d::GLM{Poisson,L}, r::Vector{Float64},
    dr::Vector{Float64}, g::Vector{Float64}) where L <: ActivationFunction

    # less efficent procedure that we implement below
    # g .= (d.x' * dr) .- (d.x' * ((dr ./ r) .* d.y))

    # second term g .= (d.x' * ((dr ./ r) .* d.y))
    d.dr .= ((dr ./ r) .* d.y) # avoid allocating a temporary for element-wise part
    mul!(g, d.x', d.dr)

    # first term and update (g .= d.x' * r -. g)
    LinearAlgebra.BLAS.gemv!('T', 1.0, d.x, r, -1.0, g)

    return g
end
# ============================================================================ #
function nlli!(d::GLM{Poisson,L}, f::ObjFlag, p::Vector{Float64}, g::StorageVec) where L <: ActivationFunction
    r, dr = predict(d, p)

    # is Optim requesting a gradient?
    if g != nothing
        dnlli!(d, r, dr, g)
    end

    # Optim requested gradient only, don't bother calculating full objective
    f == nothing && return +Inf

    return -dot(log.(r), d.y) + sum(r)
end
# ============================================================================ #
function null_nlli(d::GLM{Poisson,L}) where L <: ActivationFunction
    n = sum(d.y)
    return -n * log(n / length(d.y)) + n
end
# ============================================================================ #
@inline function dnlli!(d::GLM{Binomial,Logistic}, r::Vector{Float64},
    dr::Vector{Float64}, g::Vector{Float64}) where L <: ActivationFunction

    # as for Poisson GLM, for Binomial-Logistic we can simplify the gradient
    # compared to the case where we have an arbitrary, unknown activation
    # function (see next implmentation of dnlli!() below)

    # avoid temporary allocation
    d.dr .= d.y .- r

    # -X' (y - r)
    LinearAlgebra.BLAS.gemv!('T', -1.0, d.x, d.dr, 0.0, g)
    return g
end
# ---------------------------------------------------------------------------- #
@inline function dnlli!(d::GLM{Binomial,L}, r::Vector{Float64},
    dr::Vector{Float64}, g::Vector{Float64}) where L <: ActivationFunction

    # less efficent procedure that we implement below
    # g .= d.x' * (.-((r .- d.y) ./ (r.^2 .- r)) .* dr)

    # avoid allocating a temporary for element-wise part
    d.dr .= .-((r .- d.y) ./ (r.^2 .- r)) .* dr
    mul!(g, d.x', d.dr)

    return g
end
# ---------------------------------------------------------------------------- #
@inline function binomial_nlli(y::Vector{<:Real}, r::Vector{<:Real})
    # less efficent procedure that we implement below
    # return -(dot(log.(r), d.y) + dot(1.0 .- d.y, log.(1.0 .- r)))

    sm = 0.0
    @inbounds for k in eachindex(r)
        if y[k] > 0.0
            # positive events
            sm += log(r[k])
        else
            # NOTE: there are some cases where d.x * p yeilds large enough
            # values (> ~36.86...) that logistic(d.x * p) will yield vaules of
            # exactly 1.0 (due to round off error), so we subtract eps() to
            # avoid getting Inf from log(1.0 - 1.0), in cases where the output
            # of the activation function is 0 we're still "ok" as we just get
            # log(1 + eps())

            # negative events
            sm += log(1.0 - (r[k] - eps()))
        end
    end

    return -sm
end
# ---------------------------------------------------------------------------- #
# sources:
# * https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf
# * http://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf
function nlli!(d::GLM{Binomial,L}, f::ObjFlag, p::Vector{Float64}, g::StorageVec) where L <: ActivationFunction
    r, dr = predict(d, p)

    # is Optim requesting a gradient?
    if g != nothing
        dnlli!(d, r, dr, g)
    end

    # Optim requested gradient only, don't bother calculating full objective
    f == nothing && return +Inf

    return binomial_nlli(d.y, r)
end
# ---------------------------------------------------------------------------- #
function nlli!(d::GLM{Binomial,Logistic}, f::ObjFlag, p::Vector{Float64}, g::StorageVec, h::StorageMat)
    r, dr = predict(d, p)

    # is Optim requesting a gradient?
    if g != nothing
        dnlli!(d, r, dr, g)
    end

    # is Optim requesting a hessian?
    if h != nothing
        # h .= d.x' * (d.x .* (r .* (1.0 .- r)))
        d.dr .= (r .* (1.0 .- r))
        # h .= d.x' * (d.x .* d.dr)
        LinearAlgebra.BLAS.gemm!('T', 'N', 1.0, d.x, d.x .* d.dr, 0.0, h)
    end

    # Optim requested gradient only, don't bother calculating full objective
    f == nothing && return +Inf

    return binomial_nlli(d.y, r)
end
# ============================================================================ #
function null_nlli(d::GLM{Binomial,L}) where L <: ActivationFunction
    k = sum(d.y)
    yp = k / length(d.y)
    return -(k * log(yp) + (length(d.y) - k) * log(1.0 - yp))

    # if we normalize by the number of spikes (length(d.y)) we get:
    # ef = sum(d.y) / length(y)
    # return -(ef * log(ef) + (1.0 - ef) * log(1.0 - ef))
end
# ============================================================================ #
@inline function activate!(::Type{Identity}, xp::Vector{Float64}, dr::Vector{Float64}, bin_size::Real)
    dr .= 1.0
    return xp, dr
end
# ============================================================================ #
function activate!(::Type{Exponential}, xp::Vector{Float64}, dr::Vector{Float64}, bin_size::Real)
    # exponential is its own derivitive
    xp .= exp.(xp) .* bin_size
    return xp, xp
end
# ============================================================================ #
# see also: https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf slide 5
function activate!(::Type{Logistic}, xp::Vector{Float64}, dr::Vector{Float64}, bin_size::Real)
    # logistic function, f(xp)
    xp .= 1.0 ./ (1.0 .+ exp.(.-xp))

    # logistic derivitive, f'(xp))
    dr .= xp .* (1.0 .- xp)

    return xp, dr
end
# ============================================================================ #
function hessian(d::GLM{Poisson,Exponential}, p::Vector)
    # for computing the stderr of parameter estimates

    r, _ = predict(d, p)

    # NOTE: see also GLMspiketools/glmtools_fitting/Loss_GLM_logli.m
    # exponential is it's own derivitive (and 2nd), so the "weights" for all
    # "spiking terms" are 0 (line 101), thus the whole complicated mess just
    # becomes the following (seems crazy that it simplifies this much, but I've
    # checked the result agains Pillow's original code - for 1 example pair -
    # and it comes out correct)

    return d.x' * (d.x .* r)
end
# ============================================================================ #
function hessian(d::GLM{Binomial,Logistic}, p::Vector)
    # for computing the stderr of parameter estimates
    r, _ = predict(d, p)

    #= NOTE: see also
        https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function
        http://personal.psu.edu/jol2/course/stat597e/notes2/logit.pdf slide 19
        https://gist.github.com/faisal-w/9acdeefbddf2bbc6eb72
    =#

    return d.x' * (d.x .* (r .* (1.0 .- r)))
end
# ============================================================================ #
end

# nlli -(y log(r) + (1 - y)log(1 - r))
# d/dr = -((r - y) / (r^2 - r))
