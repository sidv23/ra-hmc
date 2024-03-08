using DrWatson
@quickactivate

using main
using Plots, Random, Distributions, ProgressMeter
using Flux, DynamicPPL, Zygote, UMAP

gr()
theme(:default)
default(fmt=:png, levels=7, lw=0.5, msw=0.5, la=0.5)
ProgressMeter.ijulia_behavior(:clear);

function plot_colorbar(cls)
    scatter([0, 0], [0, 1],
        zcolor=[0, 1], clims=(0, 1), xlims=(1, 1.1), c=cls,
        label="", colorbar_title="", framestyle=:none
    )
end

function nn_plot(theta; res=25, c=:viridis)
    x_range = collect(range(-6; stop=6, length=res))
    contourf(
        x_range, x_range,
        (x, y) -> f(theta)([x, y])[1], c=c
    )
    scatter!(
        Tuple.(eachrow(xs)),
        c=map(x -> x == 1 ? :firebrick1 : :dodgerblue1, ys),
        m=map(x -> x == 1 ? :square : :circle, ys),
        group=ys,
        legend=:bottomright
    )
end

function nn_plot_mean(thetas; F=mean, c=:viridis, res=25, bar=false)
    x_range = collect(range(-6; stop=6, length=res))
    contourf(
        x_range, x_range,
        (x, y) -> F([f(theta)([x, y])[1] for theta in eachrow(thetas)]), c=c, colorbar=bar,
    )
    scatter!(
        Tuple.(eachrow(xs)),
        c=map(x -> x == 1 ? :firebrick1 : :dodgerblue1, ys),
        m=map(x -> x == 1 ? :square : :circle, ys),
        group=ys,
        legend=:bottomright
    )
end

function make_model(ξ::DynamicPPL.Model, d)
    U(x) = min(-logjoint(ξ, (; θ=x)), floatmax(Float64))
    dU(x) = Zygote.gradient(x_ -> Zygote.forwarddiff(U, x_), x)[1]
    f(x) = max(exp(-U(x)), floatmin(Float64))
    g(x) = Zygote.gradient(x_ -> Zygote.forwarddiff(f, x_), x)
    return main.Model(ξ=ξ, d=d, f=f, g=g, U=U, dU=dU)
end

Random.seed!(2023)
s = 1.25
μ = [[1.0 1.0], [-1.0 -1.0], [1.0 -1.0], [-1.0 1.0],] .* 2
x0 = [s .* randn(30, 2) .+ μ[1]; s .* randn(80, 2) .+ μ[2]]
x1 = [s .* randn(10, 2) .+ μ[3]; s .* randn(100, 2) .+ μ[4]]
xs = [x0; x1]
ys = [zeros(size(x0, 1)); ones(size(x1, 1))] .|> Int;

scatter(Tuple.(eachrow(xs)), group=ys, ratio=1)

F = Chain(
    Dense(2, 3, sigmoid),
    Dense(3, 2, sigmoid),
    Dense(2, 1, sigmoid)
)

θ, f = Flux.destructure(F)
N = length(θ)
sigma = sqrt(1.0 / 0.09)

@model function bayes_nn(xs, ys, nθ, f)
    θ ~ MvNormal(zeros(nθ), sigma .* ones(nθ))
    F = f(θ)
    ps = F(xs)
    for i in eachindex(ys)
        ys[i] ~ Bernoulli(ps[i])
    end
end;

ξ = bayes_nn(xs', ys, N, f)
model = make_model(ξ, N);

s1, a1 = mcmc(
    DualAverage(λ=0.95, δ=0.6), RAHMC(), 
    model; n=5e3, n_burn=5e2, init=zeros(N)
)
x_rahmc = s1[a1, :];

s2, a2 = mcmc(
    DualAverage(λ=0.2, δ=0.6), HMC(),
    model; n=5e3, n_burn=5e2, init=zeros(N)
)
x_hmc = s2[a2, :];

ind = sample(1:size(x_hmc, 2), 1, replace=false)
p = []

for i in 1:size(x_hmc, 2)
    plt = histogram(x_hmc[:, i], label="HMC", fa=0.5, normalize=true)
    plt = histogram(plt, x_rahmc[:, i], label="RA-HMC", fa=0.5, normalize=true)
    title!("d = $i")
    push!(p, plt)
end

smp = collect(1:20)
plot(p[smp]..., size=(900, 900), layout=(4, 5))

X = [x_hmc; x_rahmc] |> shuffle
cls = [
    fill(:orange, size(x_hmc,1)); 
    fill(:dodgerblue, size(x_rahmc, 1))
]

embedding = umap(X', 2; n_neighbors=100, min_dist=50.1) |> eachcol .|> Tuple

scatter(embedding, c=cls, ma=0.5, label="UMAP")

cls = palette(:viridis, rev=false)

plot(
    nn_plot_mean(x_rahmc[1:5:end, :], res=20, c=cls),
    colorbar=false, title="RA-HMC",
)

cls = palette(:viridis, rev=false)

plot(
    nn_plot_mean(x_hmc[1:5:end, :], res=20, c=cls),
    colorbar=false, title="HMC",
)

plt_hmc = plot(
    nn_plot_mean(x_hmc, res=20, F=x -> quantile(x, 0.95)),
    nn_plot_mean(x_hmc, res=20, F=x -> quantile(x, 0.5)),
    nn_plot_mean(x_hmc, res=20, F=x -> quantile(x, 0.05)),
    plot_colorbar(cls),
    layout=(@layout [grid(1,3) a{0.05w}]), size=(900, 250), 
    title=["quantile " .* string.([0.9 0.5 0.1]) "HMC"], ms=1.5, lw=0.1, lc=:black
)
plt_hamram = plot(
    nn_plot_mean(x_rahmc, res=20, F=x -> quantile(x, 0.95)),
    nn_plot_mean(x_rahmc, res=20, F=x -> quantile(x, 0.5)),
    nn_plot_mean(x_rahmc, res=20, F=x -> quantile(x, 0.05)),
    plot_colorbar(cls),
    layout=(@layout [grid(1,3) a{0.05w}]), size=(900, 250), ms=1.5, lw=0.1, lc=:black,
    title=["quantile " .* string.([0.9 0.5 0.1]) "RA-HMC"]
)
plot(plt_hmc, plt_hamram, layout=(2,1), size=(800, 400))

st, res = 2, 50
cls = palette(:viridis, rev=false)

plot(
    plot(
        nn_plot_mean(x_rahmc[1:st:end, :], res=res, c=cls),
        colorbar=false, title="RA-HMC",
    ),
    plot(
        nn_plot_mean(x_hmc[1:st:end, :], res=res, c=cls),
        colorbar=false, title="HMC",
    ),
    plot_colorbar(cls),
    layout=(@layout [grid(1, 2) a{0.035w}]),
    link=:all,
    size=(1000, 400)
)
