# RAHMC
module main

using Random
using Distributions
using DistributionsAD
using Plots
using Zygote
using StatsPlots
using LinearAlgebra
using Optim
using NNlib
using Parameters
using ProgressMeter
using PDMats
using StatsBase
using Statistics
using StatsFuns
using Distances
using Setfield
using AdvancedHMC
using ForwardDiff

import Base: +, -, @kwdef


export AbstractSampler, State, PhaseState, Model
include("structs.jl")

export KE, PE, H, m2t, m2a, t2m, t2a, a2t, a2m
include("utils.jl")

export mcmc, OneStep, momentum_flip, reflect
include("mcmc.jl")

export InitializeState, RefreshState, OneStep, OnePath
export HMC, RAHMC, RAM, PEHMC, PT, DualAverage, DualAveraging
include("samplers/hmc.jl")
include("samplers/rahmc.jl")
include("samplers/ram.jl")
include("samplers/pe-hmc.jl")
include("samplers/pt.jl")
include("samplers/da.jl")

export WHMCOpt, WHMC, InitializeCache, UpdateModes, OneWHMCLeapFrog, FindInitialMode, OneWHMCStep, _posdef
include("samplers/whmc.jl")

export Plot_params, plot_pdf, contourplot, path2pts
include("plot.jl")

export Diagnostic_params, Diagnostic_plot, tracePlot, acfPlot, histPlot, diagnostics
include("diagnostics.jl")

export SteinInverseMultiquadricKernel, k, KSD
export sink_cost, W2, W2_minibatch
include("metrics.jl")


end # module main