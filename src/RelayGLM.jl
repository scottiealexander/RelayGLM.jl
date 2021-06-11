module RelayGLM

export Poisson, Binomial, Gaussian, Identity, Exponential, Logistic
export NullPrior, RidgePrior, SmoothingPrior, LassoPrior
export PredictorSet, Predictor, GLM, DefaultBasis, CosineBasis
export wasrelayed, set_parameter!, get_basis, cross_validate, get_predictor, get_basis
export get_coef, get_error, get_lambda, metric, ncol
export shuffle, JSDivergence, ROCArea, PRArea, BinomialLikelihood, RelativeBinomialLikelihood, RRI

include("./RelayUtils.jl")
using .RelayUtils

include("./GLMMetrics.jl")
using .GLMMetrics

include("./interface.jl")

import .GLMFit.GLMModels

include("./RelayISI.jl")

# ============================================================================ #
function predict(::Type{A}, ret::Vector{Float64}, xf::Vector{<:Real}, b::Real) where {A <: GLMModels.ActivationFunction}
    idx_max = round(Int, ret[end] / 0.001)
    retidx = ts2idx.(ret, 0.001)
    r, _ = GLMModels.activate(A,
        vec(
            spikefilter(retidx, reverse(xf, dims=1), (1, idx_max))
        ) .+ b,
        0.001
    )
    return r
end
# ---------------------------------------------------------------------------- #
function predict(::Type{A}, ret::Vector{Float64}, lgn::Vector{Float64}, xf::Vector{<:Real}, hf::Vector{<:Real}, b::Real) where {A <: GLMModels.ActivationFunction}
    idx_max = round(Int, max(ret[end], lgn[end]) / 0.001)

    retidx = ts2idx.(ret, 0.001)
    lgnidx = ts2idx.(lgn, 0.001)

    r, _ = GLMModels.activate(A,
        vec(
            spikefilter(retidx, reverse(xf, dims=1), (1, idx_max)) .+
            spikefilter(lgnidx, reverse(hf, dims=1), (1, idx_max))
        ) .+ b,
        0.001
    )
    return r
end
# ============================================================================ #
end
