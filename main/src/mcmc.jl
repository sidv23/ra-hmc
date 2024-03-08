function momentum_flip(state::NamedTuple)
	return (; state..., p=-state.p)
end

function reflect(state::NamedTuple, M::Model)
    du = M.dU(state.q)
    p = state.p - 2 * du * ((du'state.p) / (du'du))
    return (; state..., p=p)
end

function mcmc(S::T, M::Model; n::I=1e3, n_burn::I=1e3, init=nothing, kwargs...) where {T<:AbstractSampler,I<:Real}
    # Sample from the model M using the sampler S
    # n: number of samples
    # n_burn: number of burn-in samples

    if init === nothing
        init = randn(M.d)
    end

    state = InitializeState(init, S, M; kwargs...)

    N = Int(n + n_burn)
    samples = repeat(init', N + 1)
    accepts = fill(false, N + 1)

    p = Progress(N)
    # generate_showvalues(x) = () -> [("$(typeof(S))", x)]
    generate_showvalues(x) = () -> [("$S", x)]

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

    samples = samples[Int(n_burn):end, :]
    accepts = accepts[Int(n_burn):end]

    println("Acceptance Ratio = ", round(mean(accepts); digits=4))
    return (samples, accepts)
end