using DrWatson
@quickactivate "ra-hmc"
include(projectdir("scripts/figures/includes.jl"))

function make_model(m, s)
    m = convert(Float64, m)
    S = [[1 s; s 1], [1 -s; -s 1]]
    M = [fill(-m, 2), fill(m, 2)]
    ξ = MixtureModel([MvNormal(m, s) for (m, s) in zip(M, S)])
    return Model(ξ=ξ)
end


function plt(; l=5, length=500, kwargs...)
    xs = range(-l, l, length=length)
    contourf(xs, xs, f(model); kwargs...)
end

function scatterplot(x; p=nothing, lw=3, col=(:black, :white, :chartreuse), l="", kwargs...)
    n = round(Int, length(x) / 2 - 0.5)
    sp = isnothing(p) ? plot(0, 0, ma=0, label="") : p
    sp = plot(sp, x[1:n], c=col[1]; lw=lw, label="", kwargs...)
    sp = plot(sp, x[n:end], c=col[2]; lw=lw, label="", kwargs...)
    sp = scatter(sp, x[1], c=col[3]; label="", kwargs...)
    sp = scatter(sp, x[end], c=col[2]; label=l, kwargs...)
    return sp
end

l = 9
model = make_model(3.0, 0.75)

h(z; p=0.25) = exp(z)^p
g(model; p=0.25) = (x, y) -> h(model.f([x; y]))
f(model; p=0.25) = (x, y) -> g(model; p=p)(x, y) < 1e-1 ? -1e-1 : g(model; p=p)(x, y)

cls = cgrad(cname, rev=false, scale=:exp);
baseplt = plt(c=cls, l=l, levels=5, cb=false, axis=false, grid=false)


p1_cols = (:limegreen, :limegreen, :white)
p2_cols = (:dodgerblue1, :dodgerblue1, :white)
p3_cols = (:darkorange2, :darkorange2, :white)
p4_cols = (:darkorange2, :dodgerblue1, :white)
par_1 = (; background_color_legend=nothing, ms=7.0, msw=0.6, xlim=(-l, l), ylim=(-l, l))


z = (; q=[5.0, -3.0], p=[-0.5, -1.0], m=1.0)
p1 = @pipe z |>
           OnePath(_, HMC(ϵ=0.04, L=210), model) |>
           map(x -> Tuple(x.q), _[1]) |>
           scatterplot(_, p=baseplt, col=p1_cols, l="No friction"; par_1...)
p1 = @pipe z |>
           OnePath(_, RAHMC(ϵ=0.04, L=210, γ=-0.6), model) |>
           map(x -> Tuple(x.q), _[1][1:round(Int, end / 2)]) |>
           scatterplot(_, p=p1, col=p2_cols, l="With Friction"; par_1..., legend=:bottomright)
p1 = plot(p1, size=(300, 300))
savefig(p1, plotsdir("illustrations/attracting.pdf"))



p3_cols = (:darkorange2, :darkorange2, :white)
z = (; q=[-4.5, -4.5], p=[0.1, -1.2], m=1.0)
p2 = @pipe z |>
           OnePath(_, HMC(ϵ=0.05, L=140), model) |>
           map(x -> Tuple(x.q), _[1]) |>
           scatterplot(_, p=baseplt, col=p1_cols, l="No friction"; par_1...)
p2 = @pipe z |>
           OnePath(_, RAHMC(ϵ=0.05, L=140, γ=0.5), model) |>
           map(x -> Tuple(x.q), _[1][1:round(Int, end / 2)]) |>
           scatterplot(_, p=p2, col=p3_cols, l="Negative Friction"; par_1..., legend=:bottomright)
p2 = @pipe z |>
    OnePath(_, HMC(ϵ=0.05, L=140), model) |>
    map(x -> Tuple(x.q), _[1]) |>
    scatterplot(_, p=p2, col=p1_cols, l=""; par_1...)
p2 = plot(p2, size=(300, 300))
savefig(p2, plotsdir("illustrations/repelling.pdf"))



z = (; q=[-4.0, -3.5], p=[0.1, -0.5], m=1.0)
p3 = @pipe z |>
           OnePath(_, HMC(ϵ=0.02, L=200), model) |>
           map(x -> Tuple(x.q), _[1]) |>
           scatterplot(_, p=baseplt, col=p1_cols, l="HMC"; par_1...)
p3 = @pipe z |>
           OnePath(_, RAHMC(ϵ=0.02, L=200, γ=0.9), model) |>
           map(x -> Tuple(x.q), _[1]) |>
           scatterplot(_, p=p3, col=p4_cols, l="RA-HMC"; par_1...)
p3 = @pipe z |>
           OnePath(_, HMC(ϵ=0.02, L=200), model) |>
           map(x -> Tuple(x.q), _[1]) |>
           scatterplot(_, p=p3, col=p1_cols, l=""; par_1...)
p3 = plot(p3, size=(300, 300), legend=:bottomright)
savefig(p3, plotsdir("illustrations/rahmc.pdf"))