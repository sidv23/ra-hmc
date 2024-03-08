using Distances
######################################################################
# IMQ Kernel and Kernel gradients are taken from Gorham & Mackey (2015)
# https://github.com/jgorham/SteinDiscrepancy.jl/blob/master/src/kernels/SteinInverseMultiquadricKernel.jl

struct SteinInverseMultiquadricKernel
    beta::Float64       # -beta is exponent
    c2::Float64         # equal to c^2
    SteinInverseMultiquadricKernel(beta, c2) = (
        @assert (0 < c2);
        @assert (0 < beta);
        new(beta, c2)
    )
end

SteinInverseMultiquadricKernel(beta::Float64) = SteinInverseMultiquadricKernel(beta, 1.0)
SteinInverseMultiquadricKernel() = SteinInverseMultiquadricKernel(0.5)

function k(
    ker::SteinInverseMultiquadricKernel, 
    x::Array{Float64,1}, 
    y::Array{Float64,1})

    r = norm(x - y)
    c2 = ker.c2
    beta = ker.beta
    (c2 + r^2)^(-beta)
end

function gradxk(
    ker::SteinInverseMultiquadricKernel, 
    x::Array{Float64,1}, 
    y::Array{Float64,1})

    r = norm(x - y)
    c2 = ker.c2
    beta = ker.beta
    -2.0 * beta * (x - y) * (c2 + r^2)^(-beta - 1.0)
end

function gradxyk(
    ker::SteinInverseMultiquadricKernel, 
    x::Array{Float64,1}, 
    y::Array{Float64,1})

    r = norm(x - y)
    c2 = ker.c2
    beta = ker.beta
    d = length(x)
    2 * d * beta * (c2 + r^2)^(-beta - 1.0) - 4 * beta * (beta + 1) * r^2 * (c2 + r^2)^(-beta - 2.0)
end



######################################################################
# Stein discrepancy computed using the formula in Liu et al. (2016)
# https://arxiv.org/pdf/1602.03253.pdf


function Uq(
    x::Vector{T}, 
    y::Vector{T}; 
    params) where 
    {T<:Real}

    return (params.Sq(x)' * k(params.ker, x, y) * params.Sq(y)) +
           (params.Sq(x)' * gradxk(params.ker, y, x)) +
           (gradxk(params.ker, x, y)' * params.Sq(y)) +
           gradxyk(params.ker, x, y)
end


# TODO: Scale up for larger samples
# Takes less than 1min for samples of size ~10^3. 

function KSD(
    samples::Matrix{T}, 
    params::NamedTuple) where 
    {T<:Real}

    n = size(samples, 1)

    ksd = 0.0
    @showprogress for i in 1:n, j in 1:n
        if i != j
            ksd += Uq(samples[i, :], samples[j, :]; params=params)
        end
    end

    return ksd / (n * (n - 1))
end



######################################################################
# (Discrete) Optimal Transport Metric

function costMatrix(
    d::T, 
    x::Matrix{M}, 
    y::Matrix{M}) where 
    {M<:Real, T<:PreMetric}

    return pairwise(d, x, y, dims=1)
end


function logsinkhorn(
    C::Matrix{M}, 
    x::Matrix{M}, 
    y::Matrix{M}, 
    μ::Vector{Float64}=fill(1/size(x, 1), size(x, 1)), 
    ν::Vector{Float64}=fill(1/size(y, 1), size(y, 1)); eps::Float64=1e-1, 
    iters::T=5, 
    prog::Bool=false)::Matrix{Float64} where 
    {M<:Real,T<:Integer}


    log_P = -C ./ eps
    log_μ = log.(μ) #fill(-log(size(x, 1)), size(x, 1))
    log_ν = log.(ν)' #fill(-log(size(y, 1)), size(y, 1))'

    if prog
        @showprogress for i in 1:iters
            log_P = log_P .- (StatsFuns.logsumexp(log_P, dims=1) - log_ν)
            log_P = log_P .- (StatsFuns.logsumexp(log_P, dims=2) - log_μ)
        end
    else
        for i in 1:iters
            log_P = log_P .- (StatsFuns.logsumexp(log_P, dims=1) - log_ν)
            log_P = log_P .- (StatsFuns.logsumexp(log_P, dims=2) - log_μ)
        end
    end
    P = exp.(log_P)
    return P
end


function sink_cost(
    x::Matrix{M}, 
    y::Matrix{M},
    μ::Vector{Float64}=fill(1/size(x, 1), size(x, 1)), 
    ν::Vector{Float64}=fill(1/size(y, 1), size(y, 1)); 
    eps::Float64=1e-1, 
    iters::T=5, 
    prog::Bool=false)::Float64 where 
    {M<:Real,T<:Integer}

    
    d = SqEuclidean()
    C = costMatrix(d, x, y)
    P = logsinkhorn(C, x, y, μ, ν; eps=eps, iters=iters, prog=prog)
    return dot(P, C)
end


function W2(
    x::Matrix{M}, 
    y::Matrix{M},
    μ::Vector{Float64}=fill(1/size(x, 1), size(x, 1)), 
    ν::Vector{Float64}=fill(1/size(y, 1), size(y, 1)); 
    eps::M=1e-1, 
    iters::T=100)::Float64 where 
    {M<:Real,T<:Integer}

    return sink_cost(x, y, μ, ν; eps=eps, iters=iters, prog=true)
end

function W2_minibatch(
    x::Matrix{M}, 
    y::Matrix{M};
    μ::Vector{Float64}=fill(1/size(x, 1), size(x, 1)), 
    ν::Vector{Float64}=fill(1/size(y, 1), size(y, 1)),
    eps::M=1e-1, 
    iters::T=50, 
    k::T=100, 
    N::T=200) where 
    {M<:Real,T<:Integer}

    C = costMatrix(SqEuclidean(), x, y)
    n, m = size(C)

    P = zeros(n, m)

    @showprogress for _ in 1:k
        p = zeros(n, m)
        I = sample(1:n, N, replace=false)
        J = sample(1:m, N, replace=false)

        # C_ = costMatrix(SqEuclidean(), x[I, :], y[J, :])
        C_ = C[I, J]
        p_ = logsinkhorn(C_, x[I, :], y[J, :], μ[I], ν[J]; eps=eps, iters=iters)
        p[I, J] = p_
        P += p
    end

    dot(C, P ./ k)

end