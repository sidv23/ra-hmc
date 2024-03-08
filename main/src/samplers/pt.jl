# HMC

Base.@kwdef struct PT
    τ::Vector{R} where {R<:Real} = [1.0, 0.1]
    swap_rate::Real = 1.0
end

function InitializeState(q, pt::PT, S::AbstractSampler, M::Model; kwargs...)
    return fill(InitializeState(q, S, M), length(pt.τ))
end

function RefreshState(q, states, pt::PT, S::AbstractSampler, M::Model; kwargs...)
    return map((z, q) -> RefreshState(z.q, z, S, M), states, states)
end


function OneStep(states::Vector{Qs}, pt::PT, S::AbstractSampler, M::Model; kwargs...) where {Qs<:NamedTuple}

    results = map((z, t) -> OneStep(z, S, M; τ=t), states, pt.τ)
    newstates = map((z, r) -> rand() < r[2] ? r[1] : z, states, results)

    for _ in eachindex(pt.τ)

        if rand(Bernoulli(pt.swap_rate))
            i, j = StatsBase.sample(eachindex(pt.τ), 2, replace=false)
            Ui = M.U(newstates[i].q)
            Uj = M.U(newstates[j].q)

            mh_ratio = exp(pt.τ[i] * (Ui - Uj) + pt.τ[j] * (Uj - Ui))

            if rand() < mh_ratio
                tmp = newstates[i]
                newstates[i] = newstates[j]
                newstates[j] = tmp
            end
        end
    end

    return (newstates, results[1][2])

end


function mcmc(P::PT, S::AbstractSampler, M::Model; n::I=1e3, n_burn::I=1e3, init=nothing, kwargs...) where {I<:Real}

    if init === nothing
        init = randn(M.d)
    end

    state = InitializeState(init, P, S, M)

    N = Int(n + n_burn)
    samples = repeat(init', N + 1)
    accepts = fill(false, N + 1)

    p = Progress(N)
    generate_showvalues(x) = () -> [("$(typeof(S))", x)]

    for i ∈ 1:N
        state = RefreshState(samples[i, :], state, P, S, M; kwargs...)
        newstate, mh_ratio = OneStep(state, P, S, M; kwargs...)
        
        accept = rand() < mh_ratio
        accepts[i+1] = accept
        
        if accept
            state = newstate
            samples[i+1, :] .= newstate[1].q
        else
            samples[i+1, :] .= samples[i, :]
        end

        next!(p; showvalues=generate_showvalues(mean(accepts[1:i+1])))
    end

    samples = samples[Int(n_burn):end, :]
    accepts = accepts[Int(n_burn):end]

    println("Acceptance Ratio = ", round(mean(accepts); digits=4))
    return (samples, accepts)
end