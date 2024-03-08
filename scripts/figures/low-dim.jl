using Revise, DrWatson, Pkg
Pkg.activate("/storage/work/s/suv87/julia/ra-hmc")

begin
    using Plots
    using PyPlot
    pyplot()
end


begin
    using main
    using ProgressMeter, LinearAlgebra, Pipe
    using StatsBase, Statistics, Random
    using Distributions, Random, LinearAlgebra, Setfield, StatsBase
    using Optim, Zygote, ForwardDiff
end


settings = (; levels=5, lw=0.5, msa=0.1, msw=0.5, ma=0.2, msc=:firebrick1, legend=:topright)
settings = (; levels=5, msc=:firebrick1, legend=:topright)

# gr(; settings...)
# pgfplotsx(; settings...)
pyplot(levels=7, lw=1.0, fa=1.0, legend=:topright)
pyplot(; settings...)


function concentric(d; r1=2.0, r2=4.0, r3=6.0, s=0.25, p=2, kwargs...)
    σ = (s)^(d / 2)
    F(x) = sum([pdf(Normal(0.0, σ), norm(x, p) - r) for r in [r1; r2; r3]])
    f(x) = max(F(x), floatmin(Float64))
    U(x) = min(-log(f(x)), floatmax(Float64))
    mod = main.Model(
        ξ=MvNormal(d, 1.0),
        d=d,
        f=x -> f(x),
        g=x -> Zygote.gradient(x_ -> Zygote.forwarddiff(f, x_), x),
        U=x -> min(-log(f(x)), floatmax(Float64)),
        # dU=x -> Zygote.gradient(x_ -> Zygote.forwarddiff(U, x_), x)[1],
        dU=x -> Zygote.gradient(U, x)[1],
    )
    return mod
end


function nested(d; r1=2.0, r2=16.0, m=8.0, s=0.25, p=2, kwargs...)
    σ = (s)^(d / 2)
    M = [[-m; zeros(d - 1)], [zeros(d - 1); -m], [0; zeros(d - 1)], [zeros(d - 1); m], [m; zeros(d - 1)]]
    R = [r1; r1; r2; r1; r1]
    F(x) = sum([pdf(Normal(0.0, σ), norm(x .- m, p) - r) for (r, m) in zip(R, M)])
    f(x) = max(F(x), floatmin(Float64))
    U(x) = min(-log(f(x)), floatmax(Float64))
    mod = main.Model(
        ξ=MvNormal(d, 1.0),
        d=d,
        f=x -> f(x),
        g=x -> Zygote.gradient(x_ -> Zygote.forwarddiff(f, x_), x),
        U=x -> min(-log(f(x)), floatmax(Float64)),
        dU=x -> Zygote.gradient(x_ -> Zygote.forwarddiff(U, x_), x)[1],
    )
    return mod
end

function scatterplot(x;
    params,
    baseplt=plot(0, 0, label=""),
    label="", ma=0.05, cb=false,
    c=cgrad(:viridis, 3, categorical=true),
    kwargs...)


    nrm = norm.(eachrow(x), params.p)

    if params.type == :concentric
        cls = cgrad(:viridis, 3, categorical=true)
        plt_lim = (-params.r3, params.r3) .* 1.1
        cl = map(x -> x < params.r2 - 4 * params.s ? c[1] : (x < params.r3 - 4 * params.s ? c[2] : c[3]), nrm)
    else
        cls = cgrad(:viridis, 2, categorical=true)
        plt_lim = (-params.r3, params.r3) .* 1.1
        cl = map(x -> x < params.r2 - 4 * params.s ? 0.0 : 1.0, nrm)
    end

    plt = plot(baseplt, x |> m2t, c=:black, lw=0.1, la=0.2, label="", lim=plt_lim, ratio=1)
    plt = scatter(plt, x |> m2t, c=cl, ma=ma, label=label, cb=cb; kwargs...)
    return plt
end


begin
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
    benchmark = main.Model(ξ=ξ)
end


########################################
######## Ground truth
cls = cgrad(:linear_wcmr_100_45_c42_n256, 6, rev=false)
cname = :linear_wcmr_100_45_c42_n256

function plt(model; lim=(-5.5, 5.5), n=400, p=0.25, kwargs...)
    sq = range(lim..., length=n)
    z = [model.f([x, y])^p for y in sq, x in sq]
    # heatmap(sq, sq, z, c=cls; kwargs...)
    contourf(sq, sq, (x, y) -> model.f([x, y]) .< 0.02 ? -1e-6 : model.f([x, y])^p; kwargs...)
end


########################################
######## Benchmark
# cls = cgrad(:viridis, 10, rev=false, scale=:log)

begin
    cls = cgrad(:linear_wcmr_100_45_c42_n256, 4, rev=false)
    p = plt(benchmark, lim=(-5.5, 5.5), p=1.0, c=cls, levels=4, size=(450, 350), fa=1.0, grid=false)
end
plot(p, size=(450, 350))
savefig(plotsdir("summary/benchmark-plot.pdf"))

########################################
######## Concentric
begin
    c_params = (; type=:concentric, r1=4.0, r2=8.0, r3=12.0, s=0.5, p=1)
    cls = cgrad(:linear_wcmr_100_45_c42_n256, 7, rev=false)

    p = plt(concentric(2; c_params...), lim=(-14, 14), levels=3, n=1000, ratio=1, grid=false, p=1.0, la=0.0, c=cls)
    p = plot(p, legend=:topleft, size=(450, 350), ratio=1)
end
p
savefig(plotsdir("summary/concentric-plot.pdf"))


########################################
######## Nested
begin
    n_params = (; type=:nested, r1=4.0, r2=20.0, m=9.0, s=0.5, p=1)
    cls = cgrad(:linear_wcmr_100_45_c42_n256, 7, rev=false)

    p = plt(nested(2; n_params...), lim=(-22, 22), levels=3, n=1200, ratio=1, grid=false, p=0.95, c=cls)
    p = plot(p, legend=:topleft, size=(450, 350), ratio=1)
end
p
savefig(plotsdir("summary/nested-plot.pdf"))



########################################
######## Anisotropic Results


begin
    pgfplotsx()
    names = ["HMC" "RA-HMC" "RAM" "PEHMC"];
    dimension = [2, 10, 20, 50, 100]
    m = Any[]
    cl = palette(:tab10)[3:end]
    
    push!(m, [3.33, 5.55, 8.07, 13.11, 19.56])
    push!(m, [0.39, 0.77, 1.35, 1.99, 3.50])
    push!(m, [0.26, 5.46, 8.12, 13.50, 20.21])
    push!(m, [0.71, 3.79, 6.64, 12.41, 18.83])
    
    
    using LaTeXStrings
    using Plots.PlotMeasures
    p = plot(0, 0, ma=0, label=false, ylabel=L"\mathbf{W_{\!2}}", xlabel=L"d", legend=:topleft)
    for i in eachindex(m)
        p = plot(p, dimension, m[i], lw=2.5, label=names[i], st=:path, c=cl[i])
        p = Plots.scatter(p, dimension, m[i], lw=2.5, ma=1.0, c=cl[i], label="", msc=:black)
    end
    plot(p, legend=:topleft, size=(300, 400), bottom_margin=-4mm, left_margin=-1mm)
    savefig(plotsdir("summary/anisotropic-plot-2.pdf"))
end
pyplot()



###########

@model function funnel(d, c=1.0)
    μ = -3c
    σ = 2.0
    y ~ MixtureModel([Normal(μ + 5.0, σ), Normal(-μ - 5.0, σ)])
    x ~ MixtureModel([
        MvNormal(
            20x_ * ones(d - 1), # MEAN
            0.1 * exp(sign(x_) * 1 * (y / 2 - sign(x_) * 0.1μ + x_)) #VARIANCE
        )
        for x_ in [-c, +c]])
    return [x...; y]
end

function model(d=2, c=20.0)
    ℓ(x) = logjoint(funnel(d, c), (; x=x[1:end-1], y=x[end]))
    U(x) = isfinite(ℓ(x)) ? -ℓ(x) : 1e200
    dU(x) = ForwardDiff.gradient(U, x)
    f(x) = max(exp(-U(x)), 1e-200)
    g(x) = ForwardDiff.gradient(f, x)
    return main.Model(ξ=Normal(d, 1.0), d=d, f=f, g=g, U=U, dU=dU)
end

function plt(m=model(2), exts=[(-50, 50), (-30, 30)], len=500, cls=palette(:linear_wcmr_100_45_c42_n256, 100); levels=3, t=false, kwargs...)
    return contourf(
        range(exts[1]..., length=len),
        range(exts[2]..., length=len),
        # (x, y) -> m.f([x; y])^(2e-1), c=cls,
        (x, y) -> t ? -(m.U([x; y])) : (m.f([x; y]) .< 1e-8 ? -1e-6 : m.f([x; y])^(2e-1)),
        c=cls,
        fa=1.00,
        levels=levels; kwargs...
    )
end

c = 1.0
m = model(2, c)
pltt(l) = plt(m, ((-50, 50), (-10, 10)), size=(450, 350), t=false, levels=l, grid=false)
pltt(15)
savefig(pltt(15), plotsdir("summary/funnel.pdf"))