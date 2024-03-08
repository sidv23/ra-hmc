using Revise, DrWatson
@quickactivate

using main
using BlockDiagonals, LinearAlgebra, Pipe, Plots, ProgressMeter
using Distributions, MCMCChains, NNlib, Random, StatsBase, Zygote, ForwardDiff

gr(fmt=:png, levels=5, lw=0.5, msa=0.1, msw=0.5, ma=0.2, msc=:firebrick1, legend=:topright)
ProgressMeter.ijulia_behavior(:clear);

function generate_model(; m = 2.0, d = 10, s=0.75, method=:reverse)
    R = prod(
        [
        BlockDiagonal(
            [diagm(ones(2j)),
            [0.0 -1.0; 1.0 0.0],
            diagm(ones(d - 2j - 2))]
        ) for j in 0:round(Int, d / 2 - 1)
    ]
    )
    Σ₁ = [s^abs(i - j) for i in 1:d, j in 1:d]
    Σ₂ = R * Σ₁ * R'
    μ = [-m .* ones(d), m .* ones(d)]
    Σ = [Σ₁, Σ₂]

    ξ = MixtureModel(
        [MvNormal(x, y) for (x, y) in zip(μ, Σ)]
    )
    S = [inv(s) for s in Σ]
    
    f(x) = [-dot(x-m, s, x-m) for (m, s) in zip(μ, S)] .|> exp |> sum
    U(x) = -logsumexp([-dot(x-m, s, x-m) for (m, s) in zip(μ, S)])
    DU(x) = Zygote.gradient(U, x)[1]
    
    return Model(ξ=ξ, f=f, U=U, dU=DU)
end

function bures(m1, m2, s1, s2)
    a = norm(m1 - m2)^2
    b = tr(sqrt(sqrt(s1) * s2 * sqrt(s1)))
    c = tr(s1) + tr(s2) - 2 * b
    return sqrt(a + c)
end

function w2(X)
    if typeof(X) <: Tuple
        Y = rand(model.ξ, size(X[1], 1))' |> Matrix
        Z = X[1]
        return bures(mean(Z), mean(Y), cov(Z), cov(Y))
    else
        Y = rand(model.ξ, size(X, 1))' |> Matrix
        return bures(mean(X), mean(Y), cov(X), cov(Y))
    end
end

function scatterplot(x; baseplt=plot(0,0,label=""), label="", kwargs...)
    if typeof(x) <: Tuple
        plt = plot(baseplt, x[1] |> m2t, c=:black, lw=0.1, la=0.25, label="")
        plt = scatter(plt, x[1] |> m2t, ms=exp.(2 .* standardize(UnitRangeTransform, x[2])), label=label, c=:orange; kwargs...)
    else
        plt = plot(baseplt, x |> m2t, c=:black, lw=0.1, la=0.25, label="")
        plt = scatter(plt, x |> m2t, c=:orange, label=label; kwargs...)
    end
    return plt
end

function acfplots(chains, names, lags=0:2:50; kwargs...)
    plt = plot(0, 0)
    for (x, n) in zip(chains, names)
        plt = plot(plt, mean(1 .* (autocor(x, lags=[lags...])[:, :]), dims=1)', label=n; kwargs...)
    end
    return plt
end

function scatterplots(xs, names; baseplt=plot(0,0,label=nothing), l=400, kwargs...)
    L = length(xs)
    ds = sample(1:size(xs[1], 2), 2, replace=false)
    plts = [scatter(baseplt, x[:, ds[1]], x[:, ds[2]], label=names[i]) for (x, i) in zip(xs, eachindex(xs))]
    plot(plts..., axes=false, ticks=false; kwargs...)
end

function traceplots(xs, names, args...; baseplt=plot(0,0,label=nothing), l=400, kwargs...)
    L = length(xs)
    ds = sample(1:size(xs[1], 2), 1, replace=false)
    plts = [plot( (typeof(x) <: Tuple ? x[1][:, ds[1]] : x[:, ds[1]]), label=names[i]; args...) for (x, i) in zip(xs, eachindex(xs))]
    plot(plts...; kwargs...)
end

function mean_ess(chains)
    return [ess_rhat(chn)[:, 2] |> mean for chn in chains]
end

model = generate_model(d=2);

cls = palette(:linear_wcmr_100_45_c42_n256, 100, rev=false)
plt(; lim=(-5, 5)) = contourf(
    repeat([range(lim..., length=200)], 2)..., 
    (x, y) -> model.f([x; y]),
    c=cls
)
plt2 = plt()

@time s1, a1 = mcmc(
    DualAverage(λ=10.0, δ=0.65),
    HMC(),
    model; n=5e3, n_burn=1e3
)
x_hmc = s1[a1, :]
chain_hmc = Chains(x_hmc)
plt2_hmc = scatterplot(x_hmc[:, 1:2], baseplt=plt(), label="HMC")

@time s2, a2 = mcmc(
    DualAverage(λ=20.0, δ=0.65),
    RAHMC(),
    model; n=5e3, n_burn=1e3
)
x_rahmc = s2[a2, :]
chain_rahmc = Chains(x_rahmc)
plt2_rahmc = scatterplot(x_rahmc[:, 1:2], baseplt=plt(), label="RA-HMC")

@time s3, a3 = mcmc(
    RAM(kernel=MvNormal(model.d, 1.5), z=randn(model.d)),
    model; n=5e3, n_burn=1e3
)
x_ram = s3[a3, :]
chain_ram = Chains(x_ram)
plt2_ram = scatterplot(x_ram[:, 1:2], baseplt=plt(), label="RAM")

@time s4, w4, a4 = mcmc(
    PEHMC(ϵ=0.1, L=20, N=100),
    model; n=50, n_burn=500
)
x_pehmc, w_pehmc = [s4[a4, :]...] |> a2m, [w4[a4, :]...]
chain_pehmc = Chains(x_pehmc)
plt2_pehmc = scatterplot((x_pehmc, w_pehmc), baseplt=plt(), label="PE-HMC")

names = ("HMC", "RA-HMC", "RAM", "PE-HMC")
xs2d = [x_hmc, x_rahmc, x_ram, (x_pehmc, w_pehmc)]
chains2d = [chain_hmc, chain_rahmc, chain_ram, chain_pehmc];

plt2_tr = traceplots(xs2d, names, lw=1, layout=(2,2), ylim=(-4,4), l=100, size=(900, 500), title=["" "d=2" "" ""])

plt2_acf = acfplots(chains2d, names, lw=3)

mean_ess(chains2d)

w2.(xs2d)

model = generate_model(d=10);

@time s1, a1 = mcmc(
    DualAverage(λ=10, δ=0.6),
    HMC(),
    model; n=5e3, n_burn=1e3
)
x_10_hmc = s1[a1, :]
chain_10_hmc = Chains(x_10_hmc)
plt10_hmc = scatterplot(x_10_hmc[:, 1:2], lim=(-6, 6), label="HMC")

@time s2, a2 = mcmc(
    DualAverage(λ=20.0, δ=0.65),
    RAHMC(),
    model; n=5e3, n_burn=1e2
)
x_10_rahmc = s2[a2, :]
chain_10_rahmc = Chains(x_10_rahmc)
plt10_rahmc = scatterplot(x_10_rahmc[:, 1:2], lim=(-6, 6), label="RA-HMC")

@time s3, a3 = mcmc(
    RAM(kernel=MvNormal(model.d, 0.5), z=randn(model.d)),
    model; n=5e3, n_burn=1e3
)
x_10_ram = s3[a3, :]
chain_10_ram = Chains(x_10_ram)
plt10_ram = scatterplot(x_10_ram[:, 1:2], lim=(-6, 6), label="RAM")

@time s4, w4, a4 = mcmc(
    PEHMC(ϵ=0.08, L=15, N=100),
    model; n=5e1, n_burn=1e2
)
x_10_pehmc, w_10_pehmc = [s4[a4, :]...] |> a2m, [w4[a4, :]...]
chain_10_pehmc = Chains(x_10_pehmc)
plt10_pehmc = scatterplot((x_10_pehmc[:, 1:2], w_10_pehmc), label="PE-HMC", lim=(-6, 6))

xs10d = [x_10_hmc, x_10_rahmc, x_10_ram, (x_10_pehmc, w_10_pehmc)]
chains10d = [chain_10_hmc, chain_10_rahmc, chain_10_ram, chain_10_pehmc];

plt10_acf = acfplots(chains10d, names, lw=3)

plt10_tr = traceplots(xs10d, names, lw=1, layout=(2,2), ylim=(-5,5), l=100, size=(900, 500), title=["d=$(model.d)" "" "" ""])

mean_ess(chains10d)

w2.(xs10d)

model = generate_model(d=20);

@time s1, a1 = mcmc(
    DualAverage(λ=10, δ=0.6),
    HMC(),
    model; n=5e3, n_burn=1e3
)
x_20_hmc = s1[a1, :]
chain_20_hmc = Chains(x_20_hmc)
plt20_hmc = scatterplot(x_20_hmc[:, 1:2], label="HMC", lim=(-7, 7))

@time s2, a2 = mcmc(
    DualAverage(λ=40.0, δ=0.65),
    RAHMC(),
    model; n=5e3, n_burn=1e3
)
x_20_rahmc = s2[a2, :]
chain_20_rahmc = Chains(x_20_rahmc)
plt20_rahmc = scatterplot(x_20_rahmc[:, 1:2], label="RA-HMC", lim=(-7, 7))

@time s3, a3 = mcmc(
    RAM(kernel=MvNormal(model.d, 0.12), z=randn(model.d)),
    model; n=5e3, n_burn=1e3
)
x_20_ram = s3[a3, :]
chain_20_ram = Chains(x_20_ram)
plt20_ram = scatterplot(x_20_ram[:, 1:2], lim=(-7, 7), label="RAM")

@time s4, w4, a4 = mcmc(
    PEHMC(ϵ=0.02, L=20, N=200),
    model; n=5e1, n_burn=1e2
)
x_20_pehmc, w_20_pehmc = [s4[a4, :]...] |> a2m, [w4[a4, :]...]
chain_20_pehmc = Chains(x_20_pehmc)
plt20_pehmc = scatterplot((x_20_pehmc[:, 1:2], w_20_pehmc), label="PE-HMC", lim=(-7, 7))

xs20d = [x_20_hmc, x_20_rahmc, x_20_ram, (x_20_pehmc, w_20_pehmc)]
chains20d = [chain_20_hmc, chain_20_rahmc, chain_20_ram, chain_20_pehmc];

plt20_acf = acfplots(chains20d, names, lw=3)

plt20_tr = traceplots(xs20d, names, lw=1, layout=(2,2), ylim=(-5,5), l=100, size=(900, 500), title=["d=$(model.d)" "" "" ""])

mean_ess(chains20d)

w2.(xs20d)

model = generate_model(d=50);

@time s1, a1 = mcmc(
    DualAverage(λ=30.5, δ=0.6),
    HMC(),
    model; n=4e3, n_burn=1e3
)
x_50_hmc = s1[a1, :]
chain_50_hmc = Chains(x_50_hmc)
plt50_hmc = scatterplot(x_50_hmc[:, 1:2], label="HMC", lim=(-7, 7))

@time s2, a2 = mcmc(
    DualAverage(λ=90.0, δ=0.65),
    RAHMC(),
    model; n=4e3, n_burn=1e3
)
x_50_rahmc = s2[a2, :]
chain_50_rahmc = Chains(x_50_rahmc)
plt50_rahmc = scatterplot(x_50_rahmc[:, 1:2], label="RA-HMC", lim=(-7, 7))

@time s3, a3 = mcmc(
    RAM(kernel=MvNormal(model.d, 0.11), z=randn(model.d)),
    model; n=1e4, n_burn=1e3
)
x_50_ram = s3[a3, :]
chain_50_ram = Chains(x_50_ram)
plt50_ram = scatterplot(x_50_ram[:, 1:2], lim=(-7, 7), label="RAM")

@time s4, w4, a4 = mcmc(
    PEHMC(ϵ=0.012, L=15, N=100),
    model; n=5e1, n_burn=1e2
)
x_50_pehmc, w_50_pehmc = [s4[a4, :]...] |> a2m, [w4[a4, :]...]
chain_50_pehmc = Chains(x_50_pehmc)
plt50_pehmc = scatterplot((x_50_pehmc[:, 1:2], w_50_pehmc), label="PE-HMC", lim=(-7, 7))

xs50d = [x_50_hmc, x_50_rahmc, x_50_ram, (x_50_pehmc, w_50_pehmc)]
chains50d = [chain_50_hmc, chain_50_rahmc, chain_50_ram, chain_50_pehmc];

plt50_acf = acfplots(chains50d, names, lw=3)

plt50_tr = traceplots(xs50d, names, lw=1, layout=(2,2), ylim=(-3,3), l=100, size=(900, 500), title=["d=$(model.d)" "" "" ""])

mean_ess(chains50d)

w2.(xs50d)

model = generate_model(d=100, s=0.2);

@pipe rand(model.ξ, 5000)[1:2, :]' |> m2t |> scatterplot(_, label="ground truth", lim=(-7, 7))

@time s1, a1 = mcmc(
    DualAverage(λ=30.5, δ=0.6),
    HMC(),
    model; n=2e3, n_burn=1e3
)
x_100_hmc = s1[a1, :]
chain_100_hmc = Chains(x_100_hmc)
plt100_hmc = scatterplot(x_100_hmc[:, 1:2], label="HMC", lim=(-7, 7))

@time s2, a2 = mcmc(
    DualAverage(λ=80.0, δ=0.6),
    RAHMC(),
    model; n=2e3, n_burn=1e3
)
x_100_rahmc = s2[a2, :]
chain_100_rahmc = Chains(x_100_rahmc)
plt100_rahmc = scatterplot(x_100_rahmc[:, 1:2], label="RA-HMC", lim=(-7, 7))

@time s3, a3 = mcmc(
    RAM(kernel=MvNormal(model.d, 0.075), z=randn(model.d)),
    model; n=5e3, n_burn=1e3
)
x_100_ram = s3[a3, :]
chain_100_ram = Chains(x_100_ram)
plt100_ram = scatterplot(x_100_ram[:, 1:2], lim=(-7, 7), label="RAM")

@time s4, w4, a4 = mcmc(
    PEHMC(ϵ=0.01, L=10, N=100),
    model; n=5e1, n_burn=1e2
)
x_100_pehmc, w_100_pehmc = [s4[a4, :]...] |> a2m, [w4[a4, :]...]
chain_100_pehmc = Chains(x_100_pehmc)
plt100_pehmc = scatterplot((x_100_pehmc[:, 1:2], w_100_pehmc), label="PE-HMC", lim=(-7, 7))

xs100d = [x_100_hmc, x_100_rahmc, x_100_ram, (x_100_pehmc, w_100_pehmc)]
chains100d = [chain_100_hmc, chain_100_rahmc, chain_100_ram, chain_100_pehmc];

plt100_acf = acfplots(chains100d, names, lw=3, lags=10)

plt100_tr = traceplots(xs100d, names, lw=1, layout=(2,2), ylim=(-3,3), l=100, size=(900, 500), title=["d=$(model.d)" "" "" ""])

mean_ess(chains100d)

w2.(xs100d)
