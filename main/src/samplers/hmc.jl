# HMC

Base.@kwdef struct HMC <: AbstractSampler
    ϵ::T where {T<:Real} = 0.5
    L::Integer = 20
    cache::Any = []
end

function InitializeState(q, S::HMC, M::Model; kwargs...)
    return (; q=q, p=randn(M.d), m=1.0, kwargs...)
end

function RefreshState(q, state, S::HMC, M::Model; kwargs...)
    return (; state..., p=randn(M.d) .* sqrt.(state.m))
end

function OneStep(state::NamedTuple, S::HMC, M::Model; τ::Real=1.0, kwargs...)

    H_init = KE(state, M) + PE(state, M) * τ

    ϵ, ϵby2, L = S.ϵ, 0.5 * S.ϵ, S.L
    (; p, q, m) = state

    p = p .- (ϵby2 .* M.dU(q) .* τ)
    for _ in 1:(L-1)
        q = q .+ (ϵ .* p ./ m)
        p = p .- (ϵ .* M.dU(q) .* τ)
    end
    q = q .+ (ϵ * p ./ m)
    p = p .- (ϵby2 .* M.dU(q) .* τ)

    newstate = momentum_flip((; state..., q=q, p=p))

    H_final = KE(newstate, M) + PE(newstate, M) * τ
    mh_ratio = exp(H_init - H_final)
    # accept = rand() < mh_ratio
    # return (newstate, accept)
    return (newstate, mh_ratio)
end




function OnePath(state, S::HMC, M::Model; ref::Bool=true, τ::Real=1.0, kwargs...)

    H_init = H(state, M)

    ϵ, ϵby2, L = S.ϵ, 0.5 * S.ϵ, S.L
    (; p, q, m) = state

    k = 1
    path = fill(deepcopy(state), 1 + L)


    p = p .- (ϵby2 .* M.dU(q) .* τ)
    for _ in 1:(L-1)
        q = q .+ (ϵ .* p / m)
        p = p .- (ϵ .* M.dU(q) .* τ)
        k += 1
        path[k] = (; state..., q=q, p=p)
    end
    q = q .+ (ϵ * p / m)
    p = p .- (ϵby2 .* M.dU(q) .* τ)
    k += 1
    path[k] = (; state..., q=q, p=p)

    newstate = momentum_flip((; state..., q=q, p=p))

    H_final = H(newstate, M)
    mh_ratio = exp(H_init - H_final)
    # accept = rand() < mh_ratio
    # return (path, accept)
    return (path, mh_ratio)
end