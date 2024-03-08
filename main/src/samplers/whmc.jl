Base.@kwdef struct WHMCOpt{O <: Optim.AbstractOptimizer}
    # stepsize::Real = 0.01
    method::O = LBFGS()
    max_iter::Int = 500
    temp::Real = 1.05
    # initial_temp::Real = 1.2
    # temp_decay::Real = 0.5
    # epsilon::Real = 1e-4
end

Base.@kwdef struct WHMC <: AbstractSampler
    ϵ::T where {T<:Real} = 0.05
    L::Integer = 10
    world_distance = 1.0
    fixpoint_step = 10
    influence_factor = 1
    opt::WHMCOpt = WHMCOpt()
    k::Integer = 100
    modes::Any = []
    cache::Any = []
end

function InitializeState(q, S::WHMC, M::Model; kwargs...)
    if length(q) == M.d + 1
        return (; q=q, p=randn(M.d + 1), m=1.0, kwargs...)
    elseif length(q) > M.d + 1
        return (; q=q[1:M.d+1], p=randn(M.d + 1), m=1.0, kwargs...)
    else
        return (; q=[q...; randn(M.d + 1 - length(q))...], p=randn(M.d + 1), m=1.0, kwargs...)
    end
end

function RefreshState(q, state, S::WHMC, M::Model; kwargs...)
    return (; state..., q=q, p=randn(M.d + 1) .* sqrt.(state.m), m=1.0, kwargs...)
end

_posdef(X) = Hermitian(sqrt(Hermitian(X'X)))
function InitializeCache(S::WHMC, M::Model)
    U(q) = M.U(q[1:end-1]) + 0.5 * q[end]^2
    dU(q) = vcat(M.dU(q[1:end-1]), q[end])
    ke(z) = sum(z.p .^ 2) / (2 * z.m)
    te(z) = ke(z) + U(z.q)

    F = (; U=U, dU=dU, H=te, KE=ke)

    D2π = x -> Hermitian(pinv(_posdef(ForwardDiff.hessian(x_ -> M.U(x_), x))) + 1e-6 * I(size(x, 1)))

    if length(S.modes) == 0
        Q = MvNormal(M.d, 1.0)
    else
        Q = MixtureModel([MvNormal(m, D2π(m)) for m in S.modes])
    end
    @set! S.cache = (;
        epsby2=0.5 * S.ϵ,
        F=F,
        π=Q,
        D2π=D2π
    )
    return S
end

function UpdateModes(S::WHMC, q0::Vector{R}) where {R<:Real}
    if length(S.modes) > 0
        @set! S.cache.π = MixtureModel([S.cache.π.components; MvNormal(q0, S.cache.D2π(q0))])
    else
        @info "$(q0)"
        # @info "$(S.cache.D2π(q0))"
        @set! S.cache.π = MixtureModel([MvNormal(q0, S.cache.D2π(q0))])
    end
    push!(S.modes, q0)
    return S
end

function closest_mode(S::WHMC, q)
    dist_sq = norm.([q[1:end-1] .- p for p in S.modes])
    closest = argmin(dist_sq)
    world = q[end] < 0 ? -S.world_distance / 2 : S.world_distance / 2
    return (closest, world)
end

function mollifying_function(S::WHMC, q, k, world)
    θ_k = [S.modes[k]; world]
    θv = [[p; world] for p in S.modes]
    dist_1 = norm(q .- θ_k)
    dist_2 = norm.([q .- θ for θ in θv])
    dist_12 = norm.([θ_k .- θ for θ in θv])
    return exp.(-(dist_1 .+ dist_2 .- dist_12) / S.influence_factor)
end

function OneWHMCLeapFrog(z, z_old, S::WHMC, M::Model, ΔE, jumped, adjusted)

    p_half = z.p - S.cache.epsby2 * S.cache.F.dU(z.q)

    if jumped
        @set! z.q = z.q - S.ϵ * p_half
    else
        closest, world = closest_mode(S, z.q)
        q_m = z.q
        f_fixed = 0.0
        for jump_step in 1:S.fixpoint_step
            ms = mollifying_function(S, z_old.q, closest, world)
            if rand() < 1 - sum(ms)
                f = p_half
            else
                ps = ms ./ sum(ms)
                jump = sample(eachindex(ps), Weights(ps))

                f = ([S.modes[jump]; -world] - q_m) / S.cache.epsby2
                jumped = true
            end

            if jump_step == 1
                f_fixed = f
            end

            q_m = z.q + S.cache.epsby2 * (f_fixed + f)
        end

        closest_new, _ = closest_mode(S, q_m)

        if closest_new != closest
            ΔE += S.cache.F.U(q_m) - S.cache.F.H(z)
            adjusted = true
        end
        @set! z.q = q_m
    end

    @set! z.p = p_half - S.cache.epsby2 * S.cache.F.dU(z.q)

    if adjusted
        ΔE += S.cache.F.KE(z)
        adjusted = false
    end

    return z, ΔE, jumped, adjusted
end


# function whmc_optimize(q_old, S::WHMC, M::Model)

#     temp = S.opt.initial_temp
#     q_temp = q_old

#     tempered(temp) = x -> M.U(x) - logpdf(S.cache.π, x) / temp
#     Ur = tempered(temp)

#     for i in 1:S.opt.max_steps
#         current_grad = ForwardDiff.gradient(Ur, q_temp)

#         # @info "norm = $(norm(current_grad))"
#         if norm(current_grad) < S.opt.epsilon
#             @info "Found new mode"
#             # @info "new mode found: $q_temp"
#             S = UpdateModes(S, q_temp)
#             break
#         end

#         q_temp = q_temp - S.opt.stepsize * current_grad

#         if i % round(Int, S.opt.max_steps / 10) == 0
#             temp = temp * S.opt.temp_decay
#             Ur = tempered(temp)
#         end
#     end
#     return S
# end

function whmc_optimize(q_old, S::WHMC, M::Model)
    q_temp = q_old
    if length(S.modes) == 0
        f= x -> M.U(x)
    else
        f = x -> -log(max(1e-100, M.f(x) - exp(logpdf(S.cache.π, x) / S.opt.temp)))
    end
    result = optimize(f, q_temp; method=S.opt.method, iterations = S.opt.max_iter)
    
    if Optim.converged(result)
        S = UpdateModes(S, Optim.minimizer(result))
    end
    return S
end


function FindInitialMode(S::WHMC, M::Model)
    while S.modes == []
        @info "Finding Inital Mode..."
        S = whmc_optimize(randn(M.d), S, M)
    end
    return S
end

function OneWHMCRegen(z, S::WHMC, M::Model)
    # S.cache.π = S.π
    q_old = z.q[1:end-1]
    q_new = rand(S.cache.π)

    a = (M.f(q_new) * pdf(S.cache.π, q_old)) / (M.f(q_old) * pdf(S.cache.π, q_new))

    prob_new = pdf(S.cache.π, q_new)
    t = prob_new * a
    s = min(1, pdf(S.cache.π, q_old) / M.f(q_old))
    q = prob_new * min(1, M.f(q_new) / prob_new)

    if rand() < a
        # If regeration occurs
        if rand() < s * q / t
            S = whmc_optimize(q_old, S, M)
            prop_sample = rand(S.cache.π)

            k = 0
            while (M.f(prop_sample) <= 0.1 && k <= S.k)
                # @info "α = $(M.f(prop_sample))"
                k += 1
                prop_sample = rand(S.cache.π)
            end
            q_old = prop_sample
        else # No regeneration
            q_old = q_new
        end
    end
    @set! z.q = [q_old; z.q[end]]
    return z, S
end



function OneWHMCStep(state, S::WHMC, M::Model)
    z_old = state
    z = z_old
    ΔE, jumped, adjusted = 0, false, false
    for _ in 1:S.L
        # One leapfrog step
        z, ΔE, jumped, adjusted = OneWHMCLeapFrog(z, z_old, S, M, ΔE, jumped, adjusted)

        # One Regen Step
        z, S = OneWHMCRegen(z, S, M)
    end
    # @info "ΔE=$ΔE, H_final=$(S.cache.F.H(z)), H_init = $(S.cache.F.H(z_old)), mh_ratio=$mh_ratio"
    
    # for _ in 1:S.L
    #     z, S = OneWHMCRegen(z, S, M)
    # end
    mh_ratio = exp(-S.cache.F.H(z) + S.cache.F.H(z_old) + ΔE)

    # mh_ratio = exp(-S.cache.F.H(z) + S.cache.F.H(z_old) + ΔE)
    # @info "ΔE=$ΔE, H_final=$(S.cache.F.H(z)), H_init = $(S.cache.F.H(z_old)), mh_ratio=$mh_ratio"
    return z, mh_ratio, S
end

import main: mcmc

function mcmc(S::WHMC, M::Model; n::I=1e3, n_burn::I=1e3, init=nothing, kwargs...) where {I<:Real}

    if init === nothing
        init = randn(M.d + 1)
    end

    state = InitializeState(init, S, M; kwargs...)

    N = Int(n + n_burn)
    samples = repeat(init', N + 1)
    accepts = fill(false, N + 1)

    p = Progress(Int(N))
    generate_showvalues(x) = () -> [("$(typeof(S))", x)]
    # generate_showvalues(x) = () -> [("$(S)", x)]

    S = InitializeCache(S, M)
    if length(S.modes) == 0
        S = FindInitialMode(S, M)
    end

    for i ∈ 1:N
        state = RefreshState(samples[i, :], state, S, M; kwargs...)
        newstate, mh_ratio, S = OneWHMCStep(state, S, M; kwargs...)

        accept = rand() < mh_ratio
        accepts[i+1] = accept
        if accept
            state = newstate
            samples[i+1, :] .= newstate.q
        else
            samples[i+1, :] .= samples[i, :]
        end

        next!(p; showvalues=generate_showvalues(mean(accepts[Int(n_burn):i+1])))
    end
    samples = samples[Int(n_burn):end, 1:end-1]
    accepts = accepts[Int(n_burn):end]
    println("Acceptance Ratio = ", round(mean(accepts); digits=4))
    return samples, accepts, S
end