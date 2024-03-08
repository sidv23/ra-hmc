import Pkg
Pkg.activate("/storage/home/suv87/work/julia/ra-hmc")

using main
using Distributions, BenchmarkTools, Plots, Pipe, ProgressMeter
using Random, LinearAlgebra, Setfield, StatsBase
using Optim, Zygote, ForwardDiff

gr(fmt=:png, levels=5, lw=0.5, msa=0.1, msw=0.5, ma=0.2, msc=:firebrick1, legend=:topright)
ProgressMeter.ijulia_behavior(:clear);

params = (; r1=4.0, r2=8.0, r3=12.0, s=0.25, p=1)

# function make_model(d; r1=2.0, r2=4.0, s=0.25, m=9.0)
function make_model(d; r1=2.0, r2=4.0, r3=6.0, s=0.25, p=2)
    σ = (s)^(d / 2)
    F(x) = sum([pdf(Normal(0.0, σ), norm(x, p) - r) for r in [r1; r2; r3]])
    f(x) = max(F(x), floatmin(Float64))
    U(x) = min(-log(f(x)), floatmax(Float64))
    mod = Model(
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


function scatterplot(x; 
        baseplt=plot(0,0,label=""), 
        label="", ma=0.05, cb=false,
        c=cgrad(:viridis, 3, categorical=true), 
        kwargs...)
    
    plt_lim = (-params.r3, params.r3) .* 1.1
    
    if typeof(x) <: Tuple
        marker_size = exp.(2 .* standardize(UnitRangeTransform, x[2]))
        nrm = norm.(eachrow(x[1]), params.p)
        cl = map(x -> x < params.r2 - 4 * params.s ? c[1] : (x < params.r3 - 4 * params.s ? c[2] : c[3]), nrm)        
        
        plt = plot(baseplt, x[1] |> m2t, c=:black, lw=0.1, la=0.2, label="", lim=plt_lim, ratio=1)
        plt = scatter(plt, x[1] |> m2t, ms=marker_size, c = cl, ma=ma, label=label, cb=cb; kwargs...)
    else
        
        nrm = norm.(eachrow(x), params.p)
        cl = map(x -> x < params.r2 - 4 * params.s ? c[1] : (x < params.r3 - 4 * params.s ? c[2] : c[3]), nrm)        
        
        plt = plot(baseplt, x |> m2t, c=:black, lw=0.1, la=0.2, label="", lim=plt_lim, ratio=1)
        plt = scatter(plt, x |> m2t, c = cl, ma=ma, label=label, cb=cb; kwargs...)
    end
    return plt
end


function acfplot(x, p=1; kwargs...)
    if typeof(x) <: Tuple
        return plot(norm.(eachrow(x[1]), p); kwargs...)
    else
        return plot(norm.(eachrow(x), p); kwargs...)
    end
end

function sample_spheres(n, d; r1=2.0, r2=4.0, r3=6.0, s=0.25, p=2)
    n *= 2
    R = [r1; r2; r3]
    r = sample(R, weights(R.^(d-1)), n, replace=true) .+ rand(Normal(0.0, s), n)
    # r = sample(R, n, replace=true) .+ rand(Normal(0.0, s), n)
    X = randn(n, d)
    X = X ./ norm.(eachrow(X), p)
    return X .* r
end

function w2(X)
    if typeof(X) <: Tuple
        n, d = size(X[1])
        Z = sample_spheres(n, d; params...)
        return W2(X[1], Z, X[2] ./ sum(X[2]))
    else
        Z = sample_spheres(size(X)...; params...)
        return W2(X, Z)
    end
end

cls = palette(:linear_wcmr_100_45_c42_n256, 100, rev=false)
gr(levels=20, lw=0.5, msa=0.1, msw=0.5, ma=0.2, msc=:firebrick1, legend=:topright)
ProgressMeter.ijulia_behavior(:clear);

plt(; lim=(-14, 14), kwargs...) = contourf(
    repeat([range(lim..., length=200)], 2)...,
    (x, y) -> model.f([x, y]) ^ 0.2,
    c=cls
    ; kwargs...
)

model = make_model(2; params...);

plt(ratio=1)

@time s1, a1 = mcmc(
    DualAverage(λ=9.0, δ=0.65),
    HMC(),
    model; n=5e3, n_burn=1e3
)
x_hmc = s1[a1, :];
x_hmc |> scatterplot

@time s2, a2 = mcmc(
    DualAverage(λ=9.0, δ=0.65),
    RAHMC(),
    model; n=5e3, n_burn=1e3
)
x_rahmc = s2[a2, :];
x_rahmc |> scatterplot

@time s3, a3 = mcmc(
    RAM(kernel=MvNormal(model.d, 1.0), z=randn(model.d)),
    model; n=5e3, n_burn=1e3
)
x_ram = s3[a3, :];
x_ram |> scatterplot

@time s4, w4, a4 = mcmc(
    PEHMC(ϵ=0.05, L=20, N=100),
    model; n=5e1, n_burn=1e2
)
x_pehmc, w_pehmc = [s4[a4, :]...] |> a2m, [w4[a4, :]...];
(x_pehmc, w_pehmc) |> scatterplot

whmc_opt = WHMCOpt(method=LBFGS(), max_iter=100_000, temp=1.1)

@time s5, a5 = mcmc(
    WHMC(opt=whmc_opt, ϵ=0.2, L=10, k=1000), 
    model,
    n = 5e3, n_burn=1e3
)
x_whmc = s5[a5, :];
x_whmc |> scatterplot

var_names = ["hmc", "rahmc", "ram", "pehmc", "whmc"]
xs2d = [x_hmc, x_rahmc, x_ram, (x_pehmc, w_pehmc), x_whmc]
names = ["HMC" "RA-HMC" "RAM" "PEHMC" "WHMC"];

plot(
    scatterplot(sample_spheres(5000, 2; params...), label="ground truth", p=params.p), 
    (@pipe zip(xs2d, names) .|> scatterplot(_[1], label=_[2], p=params.p))...,
    cb=false, layout=(2, 3), size=(900, 600)
)

lims = (0.0, 1.2) .* params.r3
plot(
    (@pipe zip(xs2d, names) .|> acfplot(_[1], label=_[2]))..., 
    layout=(5, 1), size=(800, 1000), ylim=lims
)

w2.(xs2d)

model = make_model(3; params...);

scatterplot(sample_spheres(5000, 3; params...), label="ground truth", p=params.p)

d3plot(x) = @pipe x |> plot(scatterplot(_), acfplot(_, label=""), layout=(1, 2), size=(700, 400))

@time s1, a1 = mcmc(
    DualAverage(λ=5.0, δ=0.65),
    HMC(),
    model; n=5e3, n_burn=1e3
)
x_hmc_3d = s1[a1, :];
x_hmc_3d |> d3plot

@time s2, a2 = mcmc(
    DualAverage(λ=12.0, δ=0.65),
    RAHMC(), 
    model; n=5e3, n_burn=1e3
)
x_rahmc_3d = s2[a2, :];
x_rahmc_3d |> d3plot

@time s3, a3 = mcmc(
    RAM(kernel=MvNormal(model.d, 1.0), z=randn(model.d)),
    model; n=5e3, n_burn=1e3
)
x_ram_3d = s3[a3, :];
x_ram_3d |> d3plot

@time s4, w4, a4 = mcmc(
    PEHMC(ϵ=0.033, L=7, N=50),
    model; n = 1e2, n_burn=1e2
)

x_pehmc_3d, w_pehmc_3d = [s4[a4, :]...] |> a2m, [w4[a4, :]...];
(x_pehmc_3d, w_pehmc_3d) |> d3plot

@time s5, a5 = mcmc(
    WHMC(opt=whmc_opt, ϵ=0.075, L=20),
    model; n = 5e3, n_burn=1e3
)
x_whmc_3d = s5[a5, :];
@pipe x_whmc_3d |> d3plot

xs3d = [x_hmc_3d, x_rahmc_3d, x_ram_3d, (x_pehmc_3d, w_pehmc_3d), x_whmc_3d];

w2.(xs3d)
