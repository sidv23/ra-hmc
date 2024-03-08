using Revise, DrWatson
@quickactivate

using main
using BlockDiagonals, LinearAlgebra, Plots, ProgressMeter
using Distributions, MCMCChains, Random
using Optim

ProgressMeter.ijulia_behavior(:clear)
gr(fmt=:png)

μ = [
    [2.18, 5.76], [8.67, 9.59], [4.24, 8.48], [8.41, 1.68], 
    [3.93, 8.82], [3.25, 3.47], [1.70, 0.50], [4.59, 5.60], 
    [6.91, 5.81], [6.87, 5.40], [5.41, 2.65], [2.70, 7.88], 
    [4.98, 3.70], [1.14, 2.39], [8.33, 9.50], [4.93, 1.5], 
    [1.83, 0.09], [2.26, 0.31], [5.54, 6.86], [1.69, 8.11]
]

Σ = [diagm(fill(0.05, 2)) for _ in eachindex(μ)]

ξ = MixtureModel(
    [MvNormal(x .- 5.0, y) for (x, y) in zip(μ, Σ)],
    fill(1 / length(μ), length(μ))
)
model = Model(ξ=ξ);

cls = palette(:linear_wcmr_100_45_c42_n256, 100, rev=false)

gr(legendfontsize=6, levels=4, msw=0.005, lw=0.001, legend=:bottomright, axis=false, ma=0.5, msc=:firebrick1)

function plt(;lim=(-5.5, 5.5), l=200, bar=false)
    sq = range(lim..., length=l)
    contourf(sq, sq, (x, y) -> model.U([x; y]) ^ -(1+1e-1), c=cls, lw=0.1, colorbar=bar, ratio=1, grid=false)
end

function scatterplot(plt, x; kwargs...)
    p = plot(plt, x |> m2t, c=:black, lw=1, la=0.1, label="")
    p = scatter(p, x |> m2t, c=:orange; kwargs...)
end

@time s1, a1 = mcmc(
    DualAverage(λ=5.0, δ=0.65),
    HMC(),
    model; n=5e3, n_burn=1e3,
    init = zeros(2)
)
x_hmc = s1[a1, :]
plt_hmc = scatterplot(plt(), x_hmc, label="HMC")

@time s2, a2 = mcmc(
    DualAverage(λ=20.0, δ=0.65),
    RAHMC(),
    model; n=5e3, n_burn=1e3,
    init = zeros(2)
)
x_rahmc = s2[a2, :]
plt_rahmc = scatterplot(plt(), x_rahmc |> m2t, label="RA-HMC")

@time s3, a3 = mcmc(
    RAM(kernel=MvNormal(2, 2.0), z=randn(model.d)),
    model; n=5e3, n_burn=1e3,
    init = zeros(2)
)
x_ram = s3[a3, :]
plt_ram = scatterplot(plt(), x_ram |> m2t, label="RAM")

@time s4, w4, a4 = mcmc(
    PEHMC(ϵ=0.15, L=7, N=50),
    model; 
    n=1e2, n_burn=1e2,
    init=zeros(2)
)
x_pehmc, w_pehmc = [s4[a4, :]...] |> a2m, [w4[a4, :]...]
plt_pehmc = scatterplot(plt(), x_pehmc |> m2t, ms=exp.(4 .* w_pehmc), label="PEHMC")

whmc_opt = WHMCOpt(method=LBFGS(), max_iter=100_000, temp=1.2)

@time s6, a6 = mcmc(
    WHMC(opt=whmc_opt, ϵ=0.02, L=7, k=10000), model;
    n = 5e3, n_burn=1e3
)
x_whmc = s6[a6, :];
plt_whmc = scatterplot(plt(), x_whmc |> m2t, label="WHMC")

whmc_opt = WHMCOpt(method=LBFGS(), max_iter=100_000, temp=1.5)
whmc_sampler = WHMC(opt=whmc_opt, ϵ=0.02, L=7, modes=copy(μ), k=1000)

@time s5, a5 = mcmc(
    whmc_sampler, model;
    n=5e3, n_burn=1e3
)
x_whmc_known = s5[a5, :]
plt_whmc_known = scatterplot(plt(), x_whmc_known |> m2t, label="WHMC (Known modes)")

using StatsBase
acfplot(x; plt=nothing, kwargs...) = plot(plt, autocor(norm.(eachrow(x))); lw=2.0, la=1.0, kwargs...)

plt1 = acfplot(x_hmc, label="HMC", legend=:topright)
plt1 = acfplot(x_ram, plt=plt1, label="RAM")
plt1 = acfplot(x_rahmc, plt=plt1, label="RA-HMC")
plt1 = acfplot(x_pehmc, plt=plt1, label="PEHMC")
plt1 = acfplot(x_whmc, plt=plt1, label="WHMC")

function w2_minibatch(xs; eps=0.005, iters=500, k=100, N=128)
    results = zeros(length(xs))
    for (i, x) in zip(eachindex(xs), xs)
        z = Matrix(rand(model.ξ, size(x, 1))')
        results[i] = W2_minibatch(x, z, eps=eps, iters=iters, k=k, N=N)
    end
    return results
end

using Pipe
@pipe [x_hmc, x_rahmc, x_ram, x_pehmc, x_whmc] |> w2_minibatch(_)
