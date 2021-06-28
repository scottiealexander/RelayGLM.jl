module Partitions

using Random

export partition, ballanced_partition, Partitioner, IndexPartitioner,
    training_set, testing_set

abstract type AbstractPartitioner end

struct Partitioner{T<:AbstractMatrix{Float64},L<:AbstractVector{Bool}} <: AbstractPartitioner
    x::T
    y::L
    rows::Vector{Vector{Int}}
end

struct Partition
    xtrain::Matrix{Float64}
    xtest::Matrix{Float64}
    ytrain::Vector{Bool}
    ytest::Vector{Bool}
end

struct IndexPartitioner <: AbstractPartitioner
    rows::Vector{Vector{Int}}
end

IndexPartitioner(x, y, krow::Vector{Vector{Int}}) = IndexPartitioner(krow)

struct IndexPartition
    idxtrain::Vector{Int}
    idxtest::Vector{Int}
end

@inline training_set(p::Partition) = p.xtrain, p.ytrain
@inline testing_set(p::Partition) = p.xtest, p.ytest
@inline training_set(p::IndexPartition) = p.idxtrain
@inline testing_set(p::IndexPartition) = p.idxtest

Base.IteratorSize(p::AbstractPartitioner) = HasLength()
Base.IteratorEltype(p::AbstractPartitioner) = HasEltype()

@inline Base.length(p::AbstractPartitioner) = length(p.rows)
@inline Base.eltype(p::Partitioner) = Partition
@inline Base.eltype(p::IndexPartitioner) = IndexPartition

Base.iterate(p::AbstractPartitioner) = iterate(p, 1)

function Base.iterate(p::Partitioner, k::Integer)
    k > length(p.rows) && return nothing
    test = p.rows[k]
    ktrain = setdiff(1:length(p.rows), k)
    train = vcat(p.rows[ktrain]...)
    return Partition(p.x[train,:], p.x[test,:], p.y[train], p.y[test]), k + 1
end

function Base.iterate(p::IndexPartitioner, k::Integer)
    k > length(p.rows) && return nothing
    test = p.rows[k]
    ktrain = setdiff(1:length(p.rows), k)
    train = vcat(p.rows[ktrain]...)
    return IndexPartition(train, test), k + 1
end

function partition(::Type{T}, x::AbstractArray, y::AbstractArray, nfold::Integer, shfl::Bool=false) where T<:AbstractPartitioner

    size(x, 1) != length(y) && error("Sizes of inputs do not match!")

    krow = Vector{Vector{Int}}(undef, nfold)
    sz = floor(Int, length(y) / nfold)
    res = length(y) - (sz * nfold)

    idx = collect(1:length(y))
    if shfl
        shuffle!(idx)
    end

    ks = 1
    for k = 1:nfold
        ke = ks + sz
        if k > res
            ke -= 1
        end
        krow[k] = idx[ks:ke]
        ks = ke + 1
    end

    return T(x, y, krow)
end

# attempts to ballance the number of relayed spikes per test-set per fold
# such that each test set has an approximatley equal number of relayed spikes
function ballanced_partition(::Type{T}, x::AbstractArray, y::AbstractArray, nfold::Integer, shfl::Bool=false) where T<:AbstractPartitioner

    size(x, 1) != length(y) && error("Sizes of inputs do not match!")

    krow = Vector{Vector{Int}}(undef, nfold)

    kr = findall(>(0), y)
    kf = setdiff(1:length(y), kr)
    if shfl
        shuffle!(kr)
        shuffle!(kf)
    end

    nrf = floor(Int, length(kr) / nfold)
    nff = floor(Int, length(kf) / nfold)
    rres = length(kr) - (nrf * nfold)
    fres = length(kf) - (nff * nfold)

    ksr = 1
    ksf = 1
    for k = 1:nfold
        ker = ksr + nrf
        kef = ksf + nff
        if k > rres
            ker -= 1
        end
        if k > fres
            kef -= 1
        end
        krow[k] = shuffle!(vcat(kr[ksr:ker], kf[ksf:kef]))
        ksr = ker + 1
        ksf = kef + 1
    end

    return T(x, y, krow)
end

end
