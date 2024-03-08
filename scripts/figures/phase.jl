using Pipe, Plots
using Plots.Measures
using LaTeXStrings
pyplot()
cname = :linear_wcmr_100_45_c42_n256
cls = palette(cname, 10, rev=false)

function make_model(c)
    ξ = MixtureModel([Normal(x, 1.0) for x in [-c, c]])
    U(x) = -logpdf(ξ, x...)
    f(x) = pdf(ξ, x...)
    return Model(ξ=ξ, f=f, U=U)
end

function plt(; l=5, length=200, kwargs...)
    xs = range(-l, l, length=length)
    ys = range(-5, 5, length=length)
    contourf(xs, ys, f(model); kwargs...)
end

function scatterplot(x; p=nothing, col=(:black, :white, :chartreuse, :chartreuse), o=0, l=fill("", 2), kwargs...)
    n = round(Int, length(x) / 2 - o)
    sp = isnothing(p) ? plot(0, 0, ma=0, label="") : p
    sp = plot(sp, x, c=col[1]; label="", kwargs...)
    sp = scatter(sp, x[1], c=col[2]; label="", kwargs...)
    sp = scatter(sp, x[2:n], c=col[3]; label=l[1], kwargs...)
    sp = scatter(sp, x[n+1:end], c=col[4]; label=l[2], kwargs...)
    return sp
end

μ = 5.0
model = make_model(μ)

begin
    pars = (;
        levels=7,
        cb=false, grid=false, axis=false,
        bottom_margin=0mm, left_margin=0mm
    )
    cls = cgrad(cname, rev=false, scale=:exp)

    h(z) = exp(z)^0.25
    g(model) = (x, y) -> h(-model.U(x) - y^2 / 2)
    f(model) = (x, y) -> g(model)(x, y) < 1e-2 ? -1e-10 : g(model)(x, y)

    baseplt = plt(l=15, c=cls; pars...)
end


h1_cols = (:black, :white, :chartreuse, :chartreuse)
h2_cols = (:black, :white, :darkorange, :dodgerblue)
z = (; q=-1.25μ, p=1.25, m=1.0)

p1 = @pipe z |>
           OnePath(_, HMC(ϵ=0.25, L=20), model) |>
           map(x -> (x.q, x.p), _[1]) |>
           scatterplot(_, p=baseplt,
               l=["Hamiltonian Monte Carlo", ""], msw=0.2)

p1 = @pipe z |>
           OnePath(_, RAHMC(ϵ=0.25, L=10, γ=0.75), model) |>
           map(x -> (x.q, x.p), _[1]) |>
           scatterplot(_, p=p1, col=h2_cols, o=-0.5, l=["Repelling", "Attracting"], msw=0.2)


dens_pars = (; fill=0, fa=0.25, lw=2, axis=false, grid=false, label="", left_margin=-2mm, bottom_margin=-1mm)

plt2(l) = plot(model.ξ, -l, l; dens_pars...)
plt3(l=5; kwargs...) = plot([pdf(Normal(), x) for x in -l:0.01:l], -l:0.01:l; c=:firebrick, dens_pars..., kwargs...)
plt3(4)

begin
    ann = [
        (0.0, -3.0, text(L"H(q, p)", 10)),
        (0.0, 0.05, text(L"U(q)", 10)),
        (0.15, 0.0, text(L"K(p)", 10))
    ]
    adjust_p1 = (; legend=(0.0, 0.55), background_color_legend=nothing, size=(500, 500), ratio=1, legendfontsize=7, ann=ann[1])
    l = @layout [t _
        c{0.9w,0.9h} r]
    p = plot(
        plot(plt2(15), ann=ann[2], size=(800, 100)),
        plot(p1; adjust_p1..., fontsize=1),
        plot(plt3(15), yflip=true, ann=ann[3]),
        layout=l
    )
end

savefig(plotsdir("illustrations/contour.pdf"))