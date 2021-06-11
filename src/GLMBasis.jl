module GLMBasis

using LinearAlgebra

using LibPath

const LIBPATH = @libpath("libspikefilter", joinpath(@__DIR__, "lib"))

export make_stim_basis, make_spike_basis, spikefilter
# ============================================================================ #
@inline stretchx(x::Real) = log(x + 1e-20)
@inline stretchx_inv(x::Real) = exp(x) - 1e-20
@inline pimin(x::Real) = min(pi, x)
@inline pimax(x::Real) = max(-pi, x)
# ============================================================================ #
function orth(x::AbstractMatrix{<:Real})
    return try
        f = svd(x)
        k = rank(x)
        return f.U[:,1:k]
    catch err
        return fill(NaN, size(x))
    end
end
# ============================================================================ #
function get_indicies(s1::Integer, s2::Integer)
    pad = s2 - s1
    isodd(pad) && error("valid portion of result is not centered")
    n = floor(Int, pad/2)
    return n+1:s2-n
end
# ============================================================================ #
function spikefilter(idx::Vector{<:Integer}, h::AbstractArray{<:Real}, twin::Tuple{Int,Int})

    nrow, ncol = size(h)
    t1, t2 = twin
    t0 = t1 - nrow + 1
    len = (t2 - t1) + 1

    out = zeros(len, ncol)

    ks = searchsortedfirst(idx, t0)

    if ks < length(idx)

        @inbounds while (ks < length(idx)) && (idx[ks] < t2+1)
            isp = idx[ks]
            i1 = isp < t1 ? t1 : isp
            imx = isp + nrow
            imx = imx > t2 ? t2 : imx

            @inbounds for k = 1:ncol
                ycol = (k-1) * len - t1 + 1
                hcol = (k-1) * nrow + 1

                @inbounds for j = i1:(imx-1)
                    out[ycol + j] += h[hcol+j-isp]
                end

            end

            ks += 1
        end
    end

    return out
end
# ============================================================================ #
function make_basis(ncol::Integer, kpeak::AbstractVector{<:Real}, b::Real, dt::Real=0.0)
    yrange = stretchx.(kpeak .+ b)

    # it appears the original algorithm fails in some circumstances due to
    # round off error (I think...), so just use range() to guarantee the correct
    # size
    centers = range(yrange[1], yrange[2], length=ncol)
    spacing = step(centers)

    max_time = stretchx_inv(yrange[2] + 2 * spacing) - b

    if dt > 0.0
        t0 = 0.0:dt:max_time
    else
        t0 = 0:1:max_time
    end

    #this is actually faster and more efficient than the correspoding loop, not
    #sure how the compiler is pulling that off... well done Julia
    x = (repeat(stretchx.(t0 .+ b), 1, ncol) .- repeat(centers', length(t0), 1)) .* (pi / spacing / 2)

    return (cos.(pimax.(pimin.(x))) .+ 1.0) ./ 2.0, t0
end
# ============================================================================ #
function make_spike_basis(ncol::Integer, kpeak::AbstractVector{<:Real}, b::Real, dt::Real, refractory::Real=0, nt::Integer=0)

    if refractory >= dt
        ncol -= 1
    elseif refractory > 0
        @warn("Refractory period is smaller than bin size!")
    end

    basis, t0 = make_basis(ncol, kpeak, b, dt)

    if refractory >= dt
        k = findlast(x->x<=refractory, t0)
        basis[1:k, :] .= 0.0
        basis = hcat(zeros(size(basis, 1)), basis)
        basis[1:k, 1] .= 1
    end

    if nt > 0
        nrow = size(basis, 1)
        if nt > nrow
            basis = vcat(basis, zeros(nt-nrow, size(basis, 2)))
        elseif nt < nrow
            basis = basis[1:nt, :]
        end
    end

    return basis, orth(basis)
end
# ============================================================================ #
function make_stim_basis(neye::Integer, ncos::Integer, kpeak::AbstractVector{<:Integer}, b::Real, nt::Integer=0)

    basis, t0 = make_basis(ncos, kpeak, b)

    # reverse in time so that fine-scale is at the end
    basis2 = reverse(
        hcat(
            vcat(Matrix{eltype(basis)}(I, neye, neye), zeros(length(t0), neye)),
            vcat(zeros(neye, ncos), basis)
        ),
        dims=1
    )

    if nt > 0
        nrow = size(basis2, 1)
        if nt > nrow
            basis2 = vcat(zeros(nt-nrow, ncos+neye), basis2)
        elseif nt < nrow
            basis2 = basis2[end-nt+1:end, :]
        end
    end

    for k in 1:size(basis, 2)
        basis2[:,k] ./= sqrt(sum(abs2, basis[:,k]))
    end

    return basis2, orth(basis2)
end
# ============================================================================ #
end
