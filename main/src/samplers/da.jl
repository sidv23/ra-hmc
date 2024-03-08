# HMC

Base.@kwdef struct DualAverage
    δ::Real
    λ::Real = 1.0
end

function OneLeapfrog(M::Model, state::NamedTuple, ϵ::Real, τ::Real=1.0)
    (; p, q, m) = state
    ϵby2 = ϵ / 2
    p = p .- (ϵby2 .* M.dU(q) .* τ)
    q = q .+ (ϵ .* p / m)
    p = p .- (ϵby2 .* M.dU(q) .* τ)
    return (; state..., q=q, p=p)
end

function find_reasonable_epsilon(q::Vector{T}, M::Model; n_iter::Int=1_000, α::Real=0.5, eps::T=0.9) where {T <: Real}
    state = (; q=q, p=randn(M.d), m=1.0)
    ϵ, ϵ0 = fill(eps, 2)
    log_α = log(α)

    newstate = OneLeapfrog(M, state, ϵ)
    logΔH = H(state, M) - H(newstate, M)
    a = logΔH > log_α ? 1.0 : -1.0
    k = 0
    for _ in 1:n_iter
        ϵ = min(ϵ0, ϵ * α^(-a))
        newstate = OneLeapfrog(M, state, ϵ)
        logΔH = H(state, M) - H(newstate, M)
        a = logΔH > log_α ? 1.0 : -1.0
        if a * logΔH < a * log_α
            break
        end
        k += 1
    end
    return ϵ
end

function Initialize_DualVariables(S::HMC)
    @set! S.cache = (
        h=0.0,
        ϵ_dual=1.0,
        ϵ_level=max(0.0, log(10 * S.ϵ)),
        ϵ_rate=0.75,
        t0=10
    )
    return S
end

function Initialize_DualVariables(S::RAHMC)
    @set! S.cache = (
        h=0.0,
        ϵ_dual=1.0,
        γ_dual=1.0,
        ϵ_level=log(10 * S.ϵ),
        γ_level=log(10 * S.γ),
        ϵ_rate=0.75,
        γ_rate=0.75,
        t0=10
    )
    return S
end

function Update_DualVariables(S::HMC, α_MH::Real, δ::Real, m::Integer, k)
    η = m^(-k)
    @set! S.cache.h = (1 - 1 / (m + S.cache.t0)) * S.cache.h + ((δ - α_MH) / (m + S.cache.t0))
    @set! S.ϵ = exp(S.cache.ϵ_level - ((√m) / S.cache.ϵ_rate) * S.cache.h)
    @set! S.cache.ϵ_dual = exp(((1 - η) .* log(S.ϵ)) + (η .* log(S.cache.ϵ_dual)))
    return S
end

function Update_DualVariables(S::RAHMC, α_MH, δ, m, k)
    η = m^(-k)
    @set! S.cache.h = (1 - 1 / (m + S.cache.t0)) * S.cache.h + ((δ - α_MH) / (m + S.cache.t0))
    @set! S.ϵ = exp(S.cache.ϵ_level - ((√m) / S.cache.ϵ_rate) * S.cache.h)
    @set! S.γ = exp(S.cache.γ_level - ((√m) / S.cache.γ_rate) * S.cache.h)
    @set! S.cache.ϵ_dual = exp(((1 - η) .* log(S.ϵ)) + (η .* log(S.cache.ϵ_dual)))
    @set! S.cache.γ_dual = exp(((1 - η) .* log(S.γ)) + (η .* log(S.cache.γ_dual)))
    return S
end

function Set_DualVariables(S::HMC, λ)
    @set! S.ϵ = S.cache.ϵ_dual
    @set! S.L = max(4, round(Int, λ / S.ϵ))
end

function Set_DualVariables(S::RAHMC, λ)
    @set! S.ϵ = S.cache.ϵ_dual
    @set! S.γ = S.cache.γ_dual
    @set! S.L = max(4, round(Int, λ / S.ϵ))
end

function DualAveraging(D::DualAverage, S::AbstractSampler, M::Model; n_burn::Integer=100, p=nothing, ϵ0::T=1.0, kwargs...) where {T <: Real}

    @unpack λ, δ = D

    q = randn(M.d)
    state = InitializeState(q, S, M; kwargs...)

    ϵ = find_reasonable_epsilon(q, M, α=0.5)
    # ϵ = AdvancedHMC.find_good_stepsize(AdvancedHMC.Hamiltonian(DiagEuclideanMetric(M.d), M.U, ForwardDiff), q)
    @set! S.ϵ = ϵ
    
    if typeof(S) === RAHMC
        S_init, _ = DualAveraging(D, HMC(ϵ=ϵ), M; n_burn=n_burn, p=nothing)
        @set! S.ϵ = S_init.ϵ
    end
    
    S = Initialize_DualVariables(S)

    # @info "" S

    for m in 1:n_burn
        @set! S.L = round(Int, min(max(1, λ / S.ϵ), 1000))
        newstate, α_MH = OneStep(state, S, M)
        # @info "" α_MH
        α_MH = isnan(α_MH) ? 1e-10 : max(1e-10, min(1.0, α_MH))
        if rand() < α_MH
            state = (; newstate...)
        end
        S = Update_DualVariables(S, α_MH, δ, m, 0.75)
        if !isnothing(p)
            next!(p, showvalues=() -> [("$(typeof(S))", "Warming Up...")])
        end
    end
    
    S = Set_DualVariables(S, λ)    
    return (S, state)
end

function mcmc(D::DualAverage, S::AbstractSampler, M::Model; n::I=1e3, n_burn::I=1e3, init=nothing, kwargs...) where {I<:Real}

    if init === nothing
        init = randn(M.d)
    end

    state = InitializeState(init, S, M; kwargs...)

    N = Int(n)
    samples = repeat(init', N + 1)
    accepts = fill(false, N + 1)

    p = Progress(Int(N + n_burn))
    # generate_showvalues(x) = () -> [("$(typeof(S))", x)]
    generate_showvalues(x) = () -> [("$(S)", x)]

    S, state = DualAveraging(D, S, M; n_burn=Int(n_burn), p=p)
    @set! S.cache = []

    for i ∈ 1:N
        state = RefreshState(samples[i, :], state, S, M; kwargs...)
        newstate, mh_ratio = OneStep(state, S, M; kwargs...)

        accept = rand() < mh_ratio
        accepts[i+1] = accept
        if accept
            state = newstate
            samples[i+1, :] .= newstate.q
        else
            samples[i+1, :] .= samples[i, :]
        end

        next!(p; showvalues=generate_showvalues(mean(accepts[1:i+1])))
    end

    samples = samples[1:end, :]
    accepts = accepts[1:end]

    println("Acceptance Ratio = ", round(mean(accepts); digits=4))
    return (samples, accepts)
end