using Revise, DrWatson
@quickactivate

using main
using Distributions, MCMCChains, Plots, ProgressMeter

gr()
theme(:default)
default(fmt=:png, levels=7, lw=2, msw=0.5, la=0.5)
cls = [:firebrick1, :dodgerblue]
ProgressMeter.ijulia_behavior(:clear);

d = 3
model = Model(ξ=MvNormal(d, 1.0));

s1, a1 = mcmc(HMC(ϵ=0.5, L=20), model; n=5e3, n_burn=1e3);
c1_3d = Chains(s1[a1, :]);

s2, a2 = mcmc(RAHMC(ϵ=0.5, L=20, γ=0.05), model; n=5e3, n_burn=1e3);
c2_3d = Chains(s2[a2, :]);

Ds = sample(1:d, 2, replace=false)
plot(
    scatter(s1[a1, Ds] |> m2t, ma=0.1, ratio=1, lim=(-5, 5), label="HMC"),
    scatter(s2[a2, Ds] |> m2t, ma=0.1, ratio=1, lim=(-5, 5), label="RA-HMC")
)

D = sample(1:d, 1)
plot(
    begin
        histogram(s1[a1, D] |> m2t, normalize=true, label="HMC")
        plot!(x -> pdf(Normal(0, 1), x), lw=3,label="")
    end,
    begin
        histogram(s2[a2, D] |> m2t, normalize=true, label="RA-HMC")
        plot!(x -> pdf(Normal(0, 1), x), lw=3,label="")
    end,
)

plt_3 = plot([0, 0], [0, 1], c=:white, lw=0, label="d=$d")
plot!(mean(abs.(autocor(c1_3d, lags=0:20)[:, :]), dims=1)', c=cls[1], label="HMC")
plot!(mean(abs.(autocor(c2_3d, lags=0:20)[:, :]), dims=1)', c=cls[2], label="RA-HMC")

d = 10
model = Model(ξ=MvNormal(d, 1.0));

s1, a1 = mcmc(HMC(ϵ=0.5, L=20), model; n=5e3, n_burn=1e3);
c1_10d = Chains(s1[a1, :]);

s2, a2 = mcmc(RAHMC(ϵ=0.5, L=20, γ=0.05), model; n=5e3, n_burn=1e3);
c2_10d = Chains(s2[a2, :]);

plt_10 = plot([0, 0], [0, 1], c=:white, lw=0, label="d=$d")
plot!(mean(abs.(autocor(c1_10d, lags=0:20)[:, :]), dims=1)', c=cls[1], label="HMC")
plot!(mean(abs.(autocor(c2_10d, lags=0:20)[:, :]), dims=1)', c=cls[2], label="RA-HMC")

d = 50
model = Model(ξ=MvNormal(d, 1.0));

s1, a1 = mcmc(HMC(ϵ=0.5, L=20), model; n=5e3, n_burn=1e3);
c1_50d = Chains(s1[a1, :]);

s2, a2 = mcmc(RAHMC(ϵ=0.5, L=20, γ=0.05), model; n=5e3, n_burn=1e3);
c2_50d = Chains(s2[a2, :]);

plt_50 = plot([0, 0], [0, 1], c=:white, lw=0, label="d=$d")
plot!(mean(abs.(autocor(c1_50d, lags=0:20)[:, :]), dims=1)', c=cls[1], label="HMC")
plot!(mean(abs.(autocor(c2_50d, lags=0:20)[:, :]), dims=1)', c=cls[2], label="RA-HMC")

d = 100
model = Model(ξ=MvNormal(d, 1.0));

s1, a1 = mcmc(HMC(ϵ=0.5, L=20), model; n=5e3, n_burn=1e3);
c1_100d = Chains(s1[a1, :]);

s2, a2 = mcmc(RAHMC(ϵ=0.5, L=20, γ=0.05), model; n=5e3, n_burn=1e3);
c2_100d = Chains(s2[a2, :]);

plt_100 = plot([0, 0], [0, 1], c=:white, lw=0, label="d=$d")
plot!(mean(abs.(autocor(c1_100d, lags=0:20)[:, :]), dims=1)', c=cls[1], label="HMC")
plot!(mean(abs.(autocor(c2_100d, lags=0:20)[:, :]), dims=1)', c=cls[2], label="RA-HMC")

plt = plot(plt_3, plt_10, plt_50, plt_100, layout=(1, 4), size=(1000, 250))

w2(chain) = begin x = chain.value.data[:, :, 1]; W2(x, randn(size(x)...)) end
margin_of_error(d) = 5e3^(-1/d)

(; w2=w2.([c1_3d, c2_3d]), me=margin_of_error(3))

(; w2=w2.([c1_10d, c2_10d]), me=margin_of_error(10))

(; w2=w2.([c1_50d, c2_50d]), me=margin_of_error(50))

(; w2=w2.([c1_100d, c2_100d]), me=margin_of_error(100))
