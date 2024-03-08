@with_kw mutable struct Diagnostic_params
    scattercolor = :dodgerblue
    scatteralpha = 0.2
    linecolor = :dodgerblue
    linewidth = 2
    linealpha = 0.2
    densitycolor = :firebrick1
    densitywidth = 2
end;


@with_kw mutable struct Diagnostic_plot
    plt = nothing
    scatter = nothing
    trace = nothing
    hist = nothing
    acf = nothing
end;


function tracePlot(diag_params::Diagnostic_params, xsample)
    @unpack_Diagnostic_params diag_params
    return plot(xsample, label=false, title="Traceplot", c=linecolor, linewidth=linewidth, alpha=linealpha)
end




function acfPlot(diag_params::Diagnostic_params, xsample, range=1:1:20)
    @unpack_Diagnostic_params diag_params
    return plot(StatsBase.autocor(xsample, range), title="ACF Plot", label=false, c=linecolor, linewidth=linewidth, alpha=linealpha)
end




function histPlot(diag_params::Diagnostic_params, xsample, bins=50)
    @unpack_Diagnostic_params diag_params
    return D.hist = histogram(xsample, bins=bins, title="Histogram", label=false, c=linecolor, normalize=true)
end




function diagnostics(diag_params::Diagnostic_params, dim=1; M::Model, samples, bgplot)

    @unpack_Diagnostic_params diag_params

    xsample = samples[:, dim]

    D = Diagnostic_plot()

    D.scatter = bgplot()
    scatter!(D.scatter, samples[:, 1], samples[:, 2], label=false, alpha=scatteralpha, title="Scatterplot", c=scattercolor)

    D.trace = plot(xsample, label=false, title="Traceplot", c=linecolor, linewidth=linewidth, alpha=linealpha)

    D.hist = histogram(xsample, bins=50, title="Histogram", label=false, c=linecolor, normalize=true)

    D.acf = plot(StatsBase.autocor(xsample, 1:1:20), title="ACF Plot", label=false, c=linecolor, linewidth=linewidth, alpha=linealpha)

    # if M.d >= 2
    # 	D.hist2d = histogram2d(samples[1,:], samples[2,:], bins = 75, title = "2d-Histogram for first 2 components", label = false, c=linecolor, normalize=true)
    # end

    D.plt = plot(D.scatter, D.hist, D.acf, D.trace, layout=4)

    return (D)
end
