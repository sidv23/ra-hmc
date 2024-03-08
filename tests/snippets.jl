var_names = ["hmc", "rahmc", "ram", "pehmc", "whmc"]
xs2d = [x_hmc, x_rahmc, x_ram, x_pehmc, x_whmc]
names = ["HMC" "RA-HMC" "RAM" "PEHMC" "WHMC"];

for (x, n, l) in zip(xs2d, var_names, names)
    pltt = scatterplot(x, label=l, p=params.p, size=(350, 350))
    @pipe begin
        (n .* [".svg", ".pdf"]) .|>
        plotsdir("circles/concentric/scatter-2d-" * _) .|>
        savefig(pltt, _)
    end
end


#######
tmp = zip(
    [x_hmc, x_rahmc, x_ram, x_pehmc, x_whmc, x_whmc_known],
    ["hmc", "rahmc", "ram", "pehmc", "whmc", "whmc_known"],
    ["HMC" "RA-HMC" "RAM" "PEHMC" "WHMC" "WHMC (Known)"]
)

for (x, n, l) in tmp
    scatterplot(plt(), x |> m2t, label="WHMC (Known)")
    @pipe (n .* [".svg", ".pdf"]) .|>
          plotsdir("benchmark/scatter-2d-" * _) .|>
          savefig(x, _)
end




######

xs2d = [x_hmc, x_rahmc, x_ram, (x_pehmc, w_pehmc)]
var_names = ["hmc", "rahmc", "ram", "pehmc"]
names = ["HMC" "RA-HMC" "RAM" "PEHMC"];

for (x, c, n, l) in zip(xs2d, chains2d, var_names, names)
    pltt = scatterplot(x, label=l, size=(350, 350), lims=(-6, 6))
    @pipe (n .* [".svg", ".pdf"]) .|>
          plotsdir("anisotropic/scatter-2d-" * _) .|>
          savefig(pltt, _)
end

pltt = traceplots(xs2d, names, lw=1, layout=(2, 2), ylim=(-7, 7), l=100, size=(900, 500))
@pipe ("trace-2d" .* [".svg", ".pdf"]) .|> plotsdir("anisotropic/" * _) .|> savefig(pltt, _)

pltt = acfplots(chains2d, names, lw=3)
@pipe ("acf-2d" .* [".svg", ".pdf"]) .|> plotsdir("anisotropic/" * _) .|> savefig(pltt, _)



########

using DrWatson
var_names = ["hmc", "rahmc", "ram", "pehmc", "whmc"]
names = ["HMC" "RA-HMC" "RAM" "PEHMC" "WHMC"];

for (x, y, n, l) in zip(xs2d, xs3d, var_names, names)
    lims = (0.0, 1.2) .* params.r3
    plt21 = scatterplot(x, label=l, p=params.p, size=(350, 350), ma=0.25, lw=0.1)
    plt22 = acfplot(x, label=l, p=params.p, size=(800, 200), ylim=lims, legend=:bottomright)
    plt31 = scatterplot(y, label=l, p=params.p, size=(350, 350), ma=0.25, lw=0.1)
    plt32 = acfplot(y, label=l, p=params.p, size=(800, 200), ylim=lims, legend=:bottomright)
    @pipe (n .* [".svg", ".pdf"]) .|>
        plotsdir("circles/concentric/scatter-2d-" * _) .|>
        savefig(plt21, _)
    @pipe (n .* [".svg", ".pdf"]) .|>
        plotsdir("circles/concentric/trace-2d-" * _) .|>
        savefig(plt22, _)
    @pipe (n .* [".svg", ".pdf"]) .|>
        plotsdir("circles/concentric/scatter-3d-" * _) .|>
        savefig(plt31, _)
    @pipe (n .* [".svg", ".pdf"]) .|>
        plotsdir("circles/concentric/trace-3d-" * _) .|>
        savefig(plt32, _)
end