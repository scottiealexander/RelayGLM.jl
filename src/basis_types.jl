using SpkCore

include("./GLMBasis.jl")
using .GLMBasis

# ============================================================================ #
abstract type AbstractBasis end
# ============================================================================ #
struct NullBasis <: AbstractBasis
    bin_size::Float64
end
NullBasis() = NullBasis(1.0)
nbasis(n::NullBasis) = 1
project(n::NullBasis, ts::Vector{Float64}, evt::Vector{Float64}) = evt
# ============================================================================ #
mutable struct DefaultBasis <: AbstractBasis
    length::Int
    offset::Int
    bin_size::Float64
    changed::Bool
end
function DefaultBasis(;length::Integer=0, offset::Integer=0, bin_size::Real=0.001)
    @assert(length > 0, "Required length argument must be > 0")
    return DefaultBasis(length, offset, bin_size, false)
end
# ---------------------------------------------------------------------------- #
nbasis(b::DefaultBasis) = b.length
# ---------------------------------------------------------------------------- #
function project(x::DefaultBasis, ts::Vector{Float64}, evt::Vector{Float64})
    k = x.offset
    return permutedims(psth(ts, evt, -(x.length + k - 1):-k, x.bin_size)[1], (2,1))
end
# ============================================================================ #
# CosineBasis(length=100, n=4, b=4, ortho=false, bin_size=0.001)
mutable struct CosineBasis <: AbstractBasis
    length::Int
    offset::Int
    nbasis::Int
    b::Float64
    ortho::Bool
    bin_size::Float64
    basis::Matrix{Float64}
    changed::Bool
end
function CosineBasis(;length::Int=0, offset::Integer=0, nbasis::Int=8, b::Real=4.0, ortho::Bool=false, bin_size::Real=0.001)
    @assert(length > 0, "Required length argument must be > 0")
    return CosineBasis(length, offset, nbasis, b, ortho, bin_size, Matrix{Float64}(undef, 0, 0), true)
end
# ---------------------------------------------------------------------------- #
nbasis(x::CosineBasis) = x.nbasis
# ---------------------------------------------------------------------------- #
function generate(x::CosineBasis)
    if x.changed
        xpeaks = [0, round(Int, x.length * (1 - 1.5 / x.nbasis))]
        basis, ortho_basis = make_stim_basis(0, x.nbasis, xpeaks, x.b, x.length)
        if size(basis) == size(x.basis)
            x.basis .= x.ortho ? ortho_basis : basis
        else
            x.basis = x.ortho ? ortho_basis : basis
        end
        if any(isnan, x.basis)
            error("NaN detected in basis!!!")
        end
        x.changed = false
    end
    return x.basis
end
# ---------------------------------------------------------------------------- #
function project(x::CosineBasis, ts::Vector{Float64}, evt::Vector{Float64})
    idx = ts2idx.(ts, x.bin_size)
    if ts === evt
        evt_idx = idx
    else
        evt_idx = ts2idx.(evt, x.bin_size)
    end
    tidx = evt_idx .- x.offset
    if tidx[1] < 1
        filter!(>(0), tidx)
    end
    return spikefilter(idx, reverse(generate(x), dims=1), (1, evt_idx[end]))[tidx, :]
end
# ============================================================================ #
function Base.setproperty!(b::AbstractBasis, k::Symbol, v)
    setfield!(b, k, v)
    setfield!(b, :changed, true)
    return b
end
# ============================================================================ #
function set_parameter!(x::T; args...) where T <: AbstractBasis
    for (k,v) in args
        if hasfield(T, k)
            setfield!(x, k, v)
            x.changed = true
        end
    end
end
# ============================================================================ #
function Base.:*(a::Vector{<:Real},  x::CosineBasis)
    return (x.basis' * x.basis) \ (x.basis' * a)
end
Base.:*(x::CosineBasis, a::Vector{<:Real}) = x.basis * a

@inline Base.:*(a::Vector{<:Real}, x::AbstractBasis) = a
@inline Base.:*(x::AbstractBasis, a::Vector{<:Real}) = a
# ============================================================================ #
