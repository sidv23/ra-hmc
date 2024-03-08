# RAHMC

Base.@kwdef struct RAHMC <: AbstractSampler
    ϵ::T where {T<:Real} = 0.5
    L::Integer = 20
    γ::T where {T<:Real} = 0.8
    cache::Any = []
end

function InitializeState(q, S::RAHMC, M::Model; kwargs...)
    return (; q=q, p=randn(M.d), m=1.0, kwargs...)
end

function RefreshState(q, state, S::RAHMC, M::Model; kwargs...)
    return (; state..., p=randn(M.d) .* sqrt.(state.m))
end

function OneStep(state, S::RAHMC, M::Model; τ::Real=1.0, ref::Bool=true, kwargs...)

    H_init = KE(state, M) + PE(state, M) * τ

    ϵ, ϵby2, L = S.ϵ, 0.5 * S.ϵ, S.L
    β, β2 = exp(S.γ * ϵby2), exp(S.γ * ϵ)
    (; q, p, m) = state


    # Uphill Conformal Leapfrog
    p = (β * p) .- (ϵby2 * M.dU(q) .* τ)
    for i in 1:(L-1)
        q = q .+ (ϵ * p) ./ m
        p = (β2 * p) .- ((1 + β2) .* (ϵby2 * M.dU(q) .* τ))
    end
    q = q .+ (ϵ * p) ./ m
    p = β * (p - (ϵby2 * M.dU(q) .* τ))


    # change β and reflect
    β, β2 = 1 / β, 1 / β2

    # if ref
        # (; q, p) = reflect((; q=q, p=p), M)
    # end

    # Downhill Conformal Leapfrog
    p = (β * p) .- (ϵby2 * M.dU(q) .* τ)
    for i in 1:(L-1)
        q = q .+ (ϵ * p) ./ m
        p = (β2 * p) .- ((1 + β2) .* (ϵby2 * M.dU(q) .* τ))
    end
    q = q .+ (ϵ * p) ./ m
    p = β * (p - (ϵby2 * M.dU(q) .* τ))

    newstate = momentum_flip((; state..., q=q, p=p))

    H_final = KE(newstate, M) + PE(newstate, M) * τ

    mh_ratio = exp(H_init - H_final)
    # accept = rand() < mh_ratio
    return (newstate, mh_ratio)
end



function OnePath(state, S::RAHMC, M::Model; ref::Bool=true, kwargs...)

    H_init = H(state, M)

    ϵ, ϵby2, L = S.ϵ, 0.5 * S.ϵ, S.L
    β, β2 = exp(0.5 * S.γ * S.ϵ), exp(S.γ * S.ϵ)
    (; p, q, m) = state

    k = 1
    path = fill(deepcopy(state), 1 + 2L)


    # Uphill Conformal Leapfrog
    # p = (β * p) .- (ϵby2 * M.dU(q))
    # for i in 1:(L-1)
    #     q = q .+ (ϵ * p) / m
    #     p = (β2 * p) .- ((1 + β2) .* (ϵby2 * M.dU(q)))
    #     k += 1
    #     path[k] = (; state..., q=q, p=p)
    # end
    # q = q .+ (ϵ * p) / m
    # p = β * (p - (ϵby2 * M.dU(q)))
    # k += 1
    # path[k] = (; state..., q=q, p=p)
    
    for i in 1:L
        p = β * p
        p = p .- (ϵby2 * M.dU(q))
        q = q .+ (ϵ * p) ./ m
        p = p .- (ϵby2 * M.dU(q))
        p = β * p
        k += 1
        path[k] = (; state..., q=q, p=p)
    end

    # Reflect and change β
    β, β2 = 1 / β, 1 / β2

    if ref
        # tmp = reflect((; q=q, p=p), M)
        # q = tmp.q
        # p = tmp.p
    end

    # Downhill Conformal Leapfrog
    for i in 1:L
        p = β * p
        p = p .- (ϵby2 * M.dU(q))
        q = q .+ (ϵ * p) ./ m
        p = p .- (ϵby2 * M.dU(q))
        p = β * p
        k += 1
        path[k] = (; state..., q=q, p=p)
    end
    # q = q .+ (ϵ * p) / m
    # p = β * (p - (ϵby2 * M.dU(q)))
    # k += 1
    # path[k] = (; state..., q=q, p=p)

    newstate = momentum_flip((; state..., q=q, p=p))

    H_final = H(newstate, M)
    mh_ratio = exp(H_init - H_final)
    accept = rand() < mh_ratio
    return (path, accept)
end