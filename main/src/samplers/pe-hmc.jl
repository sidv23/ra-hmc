# Pseudo-Extended HMC

using NNlib

Base.@kwdef struct PEHMC <: AbstractSampler
    ϵ::T where {T<:Real} = 0.5
    L::Integer = 20
    N::Integer = 2
    cache::Any = []
end

function InitializeState(q, S::PEHMC, M::Model)
    V=(x, b) -> (NNlib.sigmoid.(b)'M.U.(x)) .+ NNlib.logsumexp((1 .- NNlib.sigmoid.(b)) .* M.U.(x) .- log(S.N))
    dV_q=(x, b) -> Zygote.gradient(x_ -> V(x_, b), x)[1]
    dV_b=(x, b) -> Zygote.gradient(b_ -> V(x, b_), b)[1]
    K=(y, a) -> sum([0.5 * (u'u + v'v) for (u, v) in zip(y, a)])
    return (;
        q=[[q]; [randn(M.d) for _ in 1:(S.N-1)]],
        b=[randn() for _ in 1:S.N],
        p=[randn(M.d) for _ in 1:S.N],
        a=[randn() for _ in 1:S.N],
        m=1.0,
        V=V,
        dV_q=dV_q,
        dV_b=dV_b,
        K=K
    )
end

function RefreshState(q, state, S::PEHMC, M::Model; kwargs...)
    return (; state..., q=q, p=[randn(M.d) .* sqrt.(state.m) for _ in 1:S.N], α=[randn() for _ in 1:S.N])
end

function OneStep(state, S::PEHMC, M::Model; ref::Bool=true, kwargs...)
    ϵ, ϵby2, L = S.ϵ, 0.5 * S.ϵ, S.L
    (; q, b, p, a, m, V, dV_q, dV_b, K) = state

    H_init = V(q, b) + K(p, a)

    p = p .- (ϵby2 .* dV_q(q, b))
    a = a .- (ϵby2 .* dV_b(q, b))
    for _ in 1:(L-1)
        q = q .+ (ϵ .* p / m)
        b = b .+ (ϵ .* a / m)
        p = p .- (ϵ .* dV_q(q, b))
        a = a .- (ϵ .* dV_b(q, b))
    end
    q = q .+ (ϵ .* p / m)
    b = b .+ (ϵ .* a / m)
    p = p .- (ϵby2 .* dV_q(q, b))
    a = a .- (ϵby2 .* dV_b(q, b))

    H_final = V(q, b) + K(p, a)
    mh_ratio = exp(H_init - H_final)
    accept = rand() < mh_ratio

    newstate = (; state..., q=q, b=b, p=-p, a=-a)
    w = NNlib.softmax((-1 .+ NNlib.sigmoid.(b)) .* M.U.(q))
    return (newstate, w, accept)
end


function mcmc(S::PEHMC, M::Model; n::I=1e3, n_burn::I=1e3, init=nothing, kwargs...) where {I<:Real}

    if init === nothing
        init = randn(M.d)
    end

    state = InitializeState(init, S, M)

    N = Int(n + n_burn)
    samples = fill(init, N + 1, S.N)
    weights = zeros(N + 1, S.N)
    accepts = fill(false, N + 1)

    p = Progress(N)
    generate_showvalues(x) = () -> [("$(typeof(S))", x)]

    for i ∈ 1:N
        state = RefreshState(samples[i, :], state, S, M; kwargs...)
        newstate, w, accept = OneStep(state, S, M; kwargs...)

        accepts[i+1] = accept
        weights[i+1, :] = w
        if accept
            state = newstate
            samples[i+1, :] .= newstate.q
        else
            samples[i+1, :] .= samples[i, :]
        end

        next!(p; showvalues=generate_showvalues(mean(accepts[1:i+1])))
    end

    samples = samples[Int(n_burn):end, :]
    weights = weights[Int(n_burn):end, :]
    accepts = accepts[Int(n_burn):end]

    println("Acceptance Ratio = ", round(mean(accepts); digits=4))
    return (samples, weights, accepts)
end