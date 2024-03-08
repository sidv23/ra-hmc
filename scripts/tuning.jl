using Revise, DrWatson
@quickactivate

using main
using Distributions, BenchmarkTools, Plots, Pipe, ProgressMeter

gr(fmt=:png, levels=5, lw=0.5, msa=0.1, msw=0.5, ma=0.2, msc=:firebrick1, legend=:topright)
ProgressMeter.ijulia_behavior(:clear);

cls = palette(:linear_wcmr_100_45_c42_n256, 100, rev=false)

function plt(d=2; lim=(-5, 5), l=200, bar=true)
    sq = range(lim..., length=l)
    if d == 2
        contourf(
            sq, sq, 
            (x, y) -> model(d).f([x; y; fill(0, model(d).d - 2)]), 
            c=cls, lw=0.1
        )
    elseif d >= 3
        contourf(
            sq, sq, 
            (x, y) -> model(d).f([x; y; fill(0, model(d).d - 2)]), 
            c=cls, lw=0.1
        )
    end
end

function make_plot(x, d, lim; kwargs...)
    k = sample(1:d, 2, replace=false)
    p = plot(plt(d; lim=lim), x -> 0, lim..., ma=0, lw=0, la=0, label="d=$d")
    # p = plot(x -> 0, lim..., ma=0, lw=0, la=0, label="d=$d")
    p = plot(p, x[:, k[1]], x[:, k[2]], c=:black, lw=0.1, la=0.25, ratio=1, label="")
    p = scatter(
        p,
        x[:, k[1]], 
        x[:, k[2]], 
        label="RA-HMC", c=:orange,
        ratio=1, grid=false; legend=:bottomright, ma=0.15
    )
end


model(d) = main.Model(
    ξ=MixtureModel(
        [MvNormal(x, 1.0 / d^0.2) for x in ([-10, +10] .* fill(ones(d)) ./ √d)]
    )
)

theme(:default)
Plots.gr_cbar_width[] = 0.02
gr(fmt=:png, levels=4, xguidefontsize=9, msc=:black, msw=0.1, tickfontsize=7)

model(d) = Model(
    ξ=MixtureModel(
        [MvNormal(x, 1.0 / d^0.2) for x in ([-10, +10] .* fill(ones(d)) ./ √d)]
    )
)

d = 3

@time s, a = mcmc(
    DualAverage(λ=15, δ=0.55),
    RAHMC(),
    model(d); n=5_000, n_burn=1_000
)

x_rahmc_3 = s[a, :];
make_plot(x_rahmc_3, d, (-8, 8))

d = 10

@time s, a = mcmc(
    DualAverage(λ=20, δ=0.55),
    RAHMC(),
    model(d); n=5_000, n_burn=1_000
)

x_rahmc_10 = s[a, :];
make_plot(x_rahmc_10, d, (-6, 6))

d = 50

@time s, a = mcmc(
    DualAverage(λ=30, δ=0.55),
    RAHMC(),
    model(d); n=5_000, n_burn=1_000
)

x_rahmc_50 = s[a, :];
make_plot(x_rahmc_50, d, (-3, 3))

d = 100

@time s, a = mcmc(
    DualAverage(λ=50, δ=0.55),
    RAHMC(),
    model(d); n=5_000, n_burn=1_000
)

x_rahmc_100 = s[a, :];
make_plot(x_rahmc_100, d, (-3, 3))
