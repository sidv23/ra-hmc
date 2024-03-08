begin
    using main
    using Pipe, Plots
    using Plots.Measures
    using LaTeXStrings
    using ProgressMeter, LinearAlgebra, Pipe
    using StatsBase, Statistics, Random
    using Distributions, Random, LinearAlgebra, Setfield, StatsBase
    using Optim, Zygote, ForwardDiff
end

pyplot()
cname = :linear_wcmr_100_45_c42_n256
cls = palette(cname, 100, rev=false);