# RAM
Base.@kwdef struct RAM <: AbstractSampler
    kernel::D where {D <: Distribution}
    z
    cache::Any = []
end

function InitializeState(q, S::RAM, M::Model)
    return (; q=q, z=randn(M.d))
end


function RefreshState(q, state, S::RAM, M::Model; kwargs...)
    return state
end


function OneStep(state, S::RAM, M::Model; kwargs...)

    q_current = state.q
    z_current = state.z

    f_q_current = M.f(q_current)
    f_z_current = M.f(z_current)


    accept = false

    ##################
    # q -> q' Downhill

    q1 = q_current + rand(S.kernel)
    while log(f_q_current + floatmin()) - log(M.f(q1) + floatmin()) < -randexp()
        q1 = q_current .+ rand(S.kernel)
    end
    f_q1 = M.f(q1)

    ##################
    # q' -> q* Uphill

    q2 = q1 + rand(S.kernel)
    while log(M.f(q2) + floatmin()) - log(f_q1 + floatmin()) < -randexp()
        q2 = q1 .+ rand(S.kernel)
    end
    f_q2 = M.f(q2)

    ##################
    # z -> z* Downhill

    z3 = q2 + rand(S.kernel)
    while log(f_q2 + floatmin()) - log(M.f(z3) + floatmin()) < -randexp()
        z3 = q2 .+ rand(S.kernel)
    end
    f_z3 = M.f(z3)

    ##################

    min_num = min(1, (f_q_current + floatmin()) / (f_z_current + floatmin()))
    min_den = min(1, (f_q2 + floatmin()) / (f_z3 + floatmin()))
    log_mh  = log(f_q2) - log(f_q_current) + log(min_num) - log(min_den)

    newstate = state
    
    if log_mh > -randexp()
        accept = true
        newstate = (; state..., q=q2, z=z3)
    end

    return (newstate, accept)
end