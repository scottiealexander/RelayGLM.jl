module RelayUtils

using SpkCore
using Statistics, Distances, Random, StatsBase

const TS = Vector{Float64}

export get_relayed, relayed_indicies, relay_status, erp, roca, precision_recall
export TS, ts2idx, ts2bin, jensen_shannon

# ============================================================================ #
function erp(ts::Vector{Float64}, evt::Vector{Float64}, pre::Real, post::Real, bin_size::Float64=0.001)
    d = ts2bin(ts, bin_size)
    evt_idx = ts2idx.(evt, bin_size)

    return erp(d, evt_idx, ts2idx(pre, bin_size), ts2idx(post, bin_size))
end
# ============================================================================ #
function erp(d::Vector{<:Real}, evt::Vector{<:Integer}, pre::Integer, post::Integer)

    len = post - pre + 1
    n = zeros(len)
    avg = fill(NaN, len, length(evt))
    k = 1
    while k <= length(evt) && evt[k] + pre < 1
        pad = abs(pre) - evt[k]
        avg[pad:end, k] .= d[1:evt[k]+post]
        k += 1
    end

    while k <= length(evt) && evt[k] + post <= length(d)
        avg[:,k] .= d[evt[k]+pre:evt[k]+post]
        k += 1
    end

    while k <= length(evt) && evt[k] + pre <= length(d)
        pad = len - ((evt[k] + post) - length(d))
        try
            avg[1:pad, k] .= d[evt[k]+pre:end]
        catch err
            @show(pad, k, evt[k], pre)
            rethrow(err)
        end
        k += 1
    end

    mn = zeros(len)
    se = zeros(len)
    for k in 1:size(avg, 1)
        tmp = filter(x->!isnan(x), avg[k,:])
        mn[k] = mean(tmp)
        se[k] = std(tmp) / sqrt(length(tmp))
    end

    return mn, se
end
# ============================================================================ #
function relay_status(ret::Vector{Float64}, lgn::Vector{Float64})
    # max_delay = 0.015
    # baseline = 0.01
    thr = 3.0
    bin_size = 0.0001

    bin = 250#round(Int, (max_delay + baseline) / bin_size)
    nbase = 100 #round(Int, baseline / bin_size)

    xcm, _ = SpkCore.psth(lgn, ret, -bin:bin, bin_size)
    xc = vec(sum(xcm, dims=2))

    _, kmax = findmax(xc)
    baseline = xc[[1:nbase; end-(nbase-1):end]]
    threshold = mean(baseline) + (std(baseline) * thr)

    kstart = kmax - (findfirst(<(threshold), view(xc, kmax:-1:1)) - 1)
    kstart = 1 <= kstart < kmax ? kstart : kmax

    kend = kmax + (findfirst(<(threshold), view(xc, kmax:length(xc))) - 1)
    kend = length(xc) >= kend > kmax ? kend : kmax

    return vec(sum(view(xcm, kstart:kend, :), dims=1))
end
# ============================================================================ #
function relayed_indicies(ret::AbstractVector{Float64}, lgn::AbstractVector{Float64})
    k_rel = findall(>(0), relay_status(ret, lgn))
    k_fail = setdiff(1:length(ret), k_rel)
    return k_rel, k_fail
end
# ============================================================================ #
function get_relayed(ret::AbstractVector{Float64}, lgn::AbstractVector{Float64}, bin_size::Real=0.0001)
    k_rel, k_fail = relayed_indicies(ret, lgn)
    return round.(Int, ret[k_rel] / bin_size) .+ 1,
        round.(Int, ret[k_fail] / bin_size) .+ 1
end
# ============================================================================ #
function phase_efficacy(ret::TS, lgn::TS, evt::TS, dur::Real, tf::Real)

    rel, fail = relayed_indicies(ret, lgn, 0.0001)

    kbin = 0:(floor(Int, dur ./ 0.001)-1)
    bpc = floor(Int, (1.0 / tf) / 0.001)
    return cycle_mean(vec(mean(psth(ret[rel], evt, kbin, 0.001)[1], dims=2)), bpc)
end
# ============================================================================ #
function roca(y::AbstractVector{<:Real}, yp::AbstractVector{<:Real})
    thr = range(ceil(maximum(yp)), stop=floor(minimum(yp)), length=100)
    hit = zeros(length(thr))
    fa = zeros(length(thr))
    acc = -Inf

    krel = findall(x->x>0, y)
    kfail = setdiff(1:length(yp), krel)

    for k in eachindex(thr)
        hit[k] = count(yp[krel] .> thr[k]) / length(krel)
        fa[k] = count(yp[kfail] .> thr[k]) / length(kfail)

        #-hits + #-correct-reject / total
        tmp = (hit[k] + (length(kfail) - fa[k])) / length(y)
        if tmp > acc
            acc = tmp
        end
    end
    return trapz(fa, hit), acc
end
# ============================================================================ #
function precision_recall(y::AbstractVector{<:Real}, yp::AbstractVector{<:Real})
    thr = range(maximum(yp), stop=floor(minimum(yp)), length=100)
    prec = zeros(length(thr))
    rec = zeros(length(thr))
    acc = -Inf

    krel = findall(>(0), y)
    kfail = setdiff(1:length(yp), krel)

    for k in eachindex(thr)
        tp = count(yp[krel] .> thr[k])
        fp = count(yp[kfail] .> thr[k])
        fn = length(krel) - tp

        # precision: true-positive / (true-positive + false-positive)
        prec[k] = tp / (tp + fp)
        prec[k] = isnan(prec[k]) ? 0.0 : prec[k]

        # recall: true-positive / (true-positive + false-negative), same as hits
        rec[k] = tp / (tp + fn)
        rec[k] = isnan(rec[k]) ? 0.0 : rec[k]

        # true-positive + true-negative / number-of-observations
        tmp = (tp + (length(kfail) - fp)) / length(y)
        if tmp > acc
            acc = tmp
        end
    end

    # see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    # "The last [first?] precision and recall values are 1. and 0. respectively
    #  and do not have a corresponding threshold. This ensures that the graph
    #  starts on the y axis."
    rec[1] = 0
    prec[1] = 1.0

    return trapz(rec, prec), acc
end
# ============================================================================ #
@enum TailType Both Left Right
# correction comes from: Phipson & Smyth 2010
function calc_pvalue(obs::Real, dist::Vector{<:Real}, tail::TailType, correction::Bool=false)
    n = length(dist) + correction
    if tail == Both
        p = (sum(x -> abs(x) >= abs(obs), dist) + correction) / n
    elseif tail == Left
        p = (sum(x -> obs >= x, dist) + correction) / n
    else
        p = (sum(x -> obs <= x, dist) + correction) / n
    end
    return p
end
# ============================================================================ #
function permute_jensen_shannon(y::AbstractVector{<:Real}, yp::AbstractVector{<:Real}, niter::Integer=1000, nbin::Integer=100)

    stat = jensen_shannon(y, yp, nbin)

    ys = copy(y)
    d = zeros(niter)
    for k in eachindex(d)
        shuffle!(ys)
        d[k] = jensen_shannon(ys, yp, nbin)
    end

    p = calc_pvalue(stat, d, Right, true)

    return stat, p
end
# ============================================================================ #
function jensen_shannon(y::AbstractVector{<:Real}, yp::AbstractVector{<:Real}, nbin::Integer=100)

    bins = range(0, 1, length=nbin)
    cr = fit(Histogram, yp[y .== 1], bins)
    cf = fit(Histogram, yp[y .== 0], bins)

    # NOTE: Distances.jl use base 'e' logarithm
    # so js_divergence maximum is log(2) ~= 0.69
    return js_divergence(cr.weights ./ trapz(cr.weights), cf.weights ./ trapz(cf.weights))
end
# ============================================================================ #
"""
trapz{T<:Real}(y::AbstractArray{T,1})

trapazoid integral approximation
"""
@inline function trapz(y::AbstractVector{<:Number})
    # return dot([0.5; ones(Float64, length(y)-2); 0.5], y)
    r = 0.5 * y[1]
    @inbounds @fastmath for k in 2:(length(y)-1)
        r += y[k]
    end
    return r + (0.5 * y[end])
end
# ============================================================================ #
"""
trapz{T<:Real}(x::AbstractArray{T,1}, y::AbstractArray{T,1})

trapazoid integral approximation of vector **y** with spacing **x**
"""
function trapz(x::AbstractVector{<:Number}, y::AbstractVector{<:Number})
    (length(x) != length(y)) && error("Inputs *MUST* be the same length!")
    r = 0.0
    @inbounds @fastmath for k = 2:length(y)
        r += (x[k] - x[k-1]) * (y[k] + y[k-1])
    end
    return r/2.0
end
# ============================================================================ #
ts2idx(ts::Float64, bs::Real=0.001) = round(Int, ts / bs) + 1
ts2bin(ts::Vector{Float64}, bs::Real=0.001) = ts2bin(Int8, ts, bs)
function ts2bin(::Type{T}, ts::Vector{Float64}, bs::Real=0.001) where T<:Real
    idx = ts2idx.(ts, bs)
    d = zeros(T, idx[end])
    d[idx] .= T(1)
    return d
end
# ============================================================================ #
end
