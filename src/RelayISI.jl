module RelayISI

using SpkCore
using Statistics, Random, StatsBase, Optim, ImageFiltering

using LoopVectorization

using ..RelayUtils, ..GLMMetrics, ..GLMFit.Partitions
import ..PredictorSet, ..Predictor, ..GLM, ..cross_validate, ..NullPrior,
       ..Binomial, ..Logistic, ..wasrelayed

export isi_cross_validate

# ============================================================================ #
function spike_status(ret::Vector{Float64}, lgn::Vector{Float64})

    status = wasrelayed(ret, lgn)
    # first spike has no isi so we can't make a prediction...
    return diff(ret), status[2:end]
end
# ============================================================================ #
function smooth_ef(ef::AbstractVector{<:Real}, sigma::Real, bin_size::Real)
    return imfilter(ef, (KernelFactors.IIRGaussian(sigma/bin_size),))
end
# ============================================================================ #
function get_eff(isi::Vector{Float64}, status::AbstractVector{<:Real}, kuse::AbstractVector{<:Integer},
        sigma::Real, isibin::Real, isimax::Real)

        isi_use = isi[kuse]
        status_use = status[kuse]
        krel = findall(>(0), status_use)

        edges = 0.002:isibin:isimax

        all = fit(Histogram, isi_use, edges)
        rel = fit(Histogram, isi_use[krel], edges)

        eff = Vector{Float64}(undef, length(all.weights))
        @turbo thread=8 for k in eachindex(all.weights)
            eff[k] = all.weights[k] > 0 ? rel.weights[k] / all.weights[k] : 0.0
        end

        if sigma > 0.0
            out = smooth_ef(eff, sigma, isibin)
        else
            out = eff
        end

        # ensure [0,1]
        @turbo thread=8 for k in eachindex(out)
            out[k] = out[k] < 0.0 ? 0.0 : out[k]
        end

    return edges, out
end
# ============================================================================ #
@inline function predict(edges::AbstractVector{<:Real}, ef::AbstractVector{T}, isi::Vector{Float64}) where T
    return predict!(fill(T(0), length(isi)), edges, ef, isi)
end
# ---------------------------------------------------------------------------- #
function predict!(p::Vector{T}, edges::AbstractVector{<:Real}, ef::AbstractVector{T}, isi::Vector{Float64}) where T<:Real

    mu = mean(ef)

    Threads.@threads for k in eachindex(p)
        @inbounds begin
            x = isi[k]
            if x >= edges[end]
                # for all isi's beyond isimax (edges[end]) predict the mean
                # efficacy mu
                p[k] = mu
            elseif x < edges[1]
                p[k] = ef[1]
            else
                j = findfirst(>(x), edges)
                # subtract 1 as we used > above to locate the upper edge of the
                # bin that this isi belongs in
                p[k] = ef[j-1]
            end
        end
    end
    return p
end
# ============================================================================ #
roundn(x::Real, n::Integer) = round(x / 10.0^n) * 10.0^n
# ============================================================================ #
function isi_cross_validate(::Type{T}, ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real},
    isimax::Vector{Float64}, isibin::Real=0.001, nfold::Integer=10) where T <: PerformanceMetric

    sigma = roundn.(vcat(0.0, 10 .^ range(log10(0.002), log10(0.03), length=7)), -3)

    out = T(nfold)
    sig = 0.0
    isimx = 0.0
    best = worst_value(T)

    isi, status = spike_status(ret, lgn)

    for row in eachrow([repeat(sigma, inner=length(isimax)) repeat(isimax, outer=length(sigma))])
        res = isi_model(T, isi, status, row[1], isibin, row[2], nfold, true)
        tmp = mean(res)[1]
        if better_than(T, tmp, best)
            copy!(out, res)
            sig = row[1]
            isimx = row[2]
            best = tmp
        end
    end

    return out, sig, isimx
end
# ============================================================================ #
# NOTE: we are NOT using the turbo functions from RelayUtils as they do not work
# with ForwardDiff.Dual types [which we need to get the objective gradient from
# ForwardDiff (via Optim) in scale_ef()]; however, Dual numbers work just fine
# with Threads (obviously...)
function logistic!(x::AbstractVector{<:Real})
    Threads.@threads for k in eachindex(x)
        @inbounds x[k] = 1.0 / (1.0 + exp(-x[k]))
    end
    return x
end
# ============================================================================ #
# NOTE: this is combined logistic + NEGATIVE likelihood calculation specifically
# for scale_ef() below
function binomial_logistic_nlli!(x::Vector{<:Real}, status::Vector{<:Real})
    Threads.@threads for k in eachindex(status)
        @inbounds begin
            tmp = 1.0 / (1.0 + exp(-x[k]))
            x[k] = status[k] > 0 ? log(tmp + eps()) : log(1.0 - (tmp - eps()))
        end
    end
    return -sum(x)
end
# ============================================================================ #
function scale_ef(edges::AbstractVector{<:Real}, ef::Vector{Float64}, isi::Vector{Float64}, status::AbstractVector{<:Real}, isibin::Real)

    cache = Dict{DataType, Any}(Float64 => zeros(length(isi)))
    N = length(isi)

    objective(p::Vector{T}) where T =  begin

        yp = get!(cache, T) do
            Vector{T}(undef, N)
        end::Vector{T}

        return binomial_logistic_nlli!(predict!(yp, edges, p[1] .+ ef .* p[2], isi), status)
    end

    x0 = [0.0, 1.0]

    # mn = mean(ef)
    # mx = maximum(ef)
    # x0 = [-mn, 1.0/(mx - mn)]

    res = optimize(objective, x0, LBFGS(); autodiff=:forward)

    return res.minimizer[1] .+ ef .* res.minimizer[2]
end
# ============================================================================ #
function isi_model(::Type{T}, isi::AbstractVector{<:Real}, status::Vector{Bool},
    sigma::Real, isibin::Real, isimax::Real, nfold::Integer, shfl::Bool) where T <: PerformanceMetric

    res = T(nfold)

    k = 1
    for p in ballanced_partition(IndexPartitioner, isi, status, nfold, shfl)
        idxtrain = training_set(p)
        idxtest = testing_set(p)

        edges, ef = get_eff(isi, status, idxtrain, sigma, isibin, isimax)
        ef = scale_ef(edges, ef, isi[idxtrain], status[idxtrain], isibin)

        pred = logistic!(predict(edges, ef, isi[idxtest]))

        eval_and_store!(res, status[idxtest], pred, k)

        k += 1
    end

    return res
end
# ============================================================================ #
end
