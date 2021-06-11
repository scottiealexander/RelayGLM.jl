module RelayISI

using SpkCore
using Statistics, Random, StatsBase, Optim, ImageFiltering

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

        eff = zeros(length(all.weights))
        @inbounds for k in eachindex(all.weights)
            if all.weights[k] > 0
                eff[k] = rel.weights[k] / all.weights[k]
            end
        end

        if sigma > 0.0
            out = smooth_ef(eff, sigma, isibin)
        else
            out = eff
        end

        # ensure [0,1]
        @inbounds for k in eachindex(out)
            if out[k] < 0.0
                out[k] = 0.0
            end
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
function logistic!(x::AbstractVector{<:Real})
    x .= 1.0 ./ (1.0 .+ exp.(.-x))
    return x
end
# ============================================================================ #
function binomial_nlli!(yp::Vector{<:Real}, status::Vector{<:Real})
    Threads.@threads for k in eachindex(status)
        @inbounds begin
            if status[k] > 0
                yp[k] = log(yp[k] + eps())
            else
                yp[k] = log(1.0 - (yp[k] - eps()))
            end
        end
    end
    return -sum(yp)
end
# ============================================================================ #
function scale_ef(edges::AbstractVector{<:Real}, ef::Vector{Float64}, isi::Vector{Float64}, status::AbstractVector{<:Real}, isibin::Real)

    cache = Dict{DataType, Any}(Float64 => zeros(length(isi)))
    N = length(isi)

    objective(p::Vector{T}) where T =  begin

        yp = get!(cache, T) do
            Vector{T}(undef, N)
        end::Vector{T}

        logistic!(predict!(yp, edges, p[1] .+ ef .* p[2], isi))

        return binomial_nlli!(yp, status)
    end

    mn = mean(ef)
    mx = maximum(ef)
    res = optimize(objective, [-mn, 1.0/(mx - mn)], LBFGS(); autodiff=:forward)

    return res.minimizer[1] .+ ef .* res.minimizer[2]
end
# ============================================================================ #
function isi_model(::Type{T}, isi::AbstractVector{<:Real}, status::AbstractVector{<:Real},
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
