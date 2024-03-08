@with_kw mutable struct Plot_params
    quiver = true
    plotband = 5.0
    quiverband = 0.8 * 5.0
    numquiver = 11
    scattercolor = "purple"
    contourcolor = :magma
    contourFn = contourf
    quivercolor = :white
    quiveralpha = 0.2
    legend = :none
    colorscheme = :inferno
    rev = true
end;

function plot_pdf(model::Model, plot_params::Plot_params; type="heatmap")

    # @unpack quiver, plotband, quiverband, numquiver = plot_params
    @unpack_Plot_params plot_params

    cls = cgrad(colorscheme, rev=rev, categorical=false)

    xs = range(-quiverband, stop=quiverband, length=11)
    ys = range(-quiverband, stop=quiverband, length=11)

    if type == "contourfn"
        plt = contourFn(-plotband:0.05:plotband, -plotband:0.05:plotband,
            (x, y) -> pdf(model.ξ, [x, y]), c=cls, aspect_ratio=:equal, legend=legend, levels=5)
    elseif type == "heatmap"
        plt = heatmap(-plotband:0.1:plotband, -plotband:0.1:plotband,
            (x, y) -> pdf(model.ξ, [x, y]), c=cls, aspect_ratio=:equal, legend=legend)
    elseif type == "contour"
        plt = StatsPlots.contour(-plotband:0.1:plotband, -plotband:0.1:plotband,
            (x, y) -> pdf(model.ξ, [x, y]), c=cls, aspect_ratio=:equal, legend=legend, levels=20)
    end


    # quiver!(repeat(xs, 11), vec(repeat(ys', 11)),
    #     quiver = (x, y) -> 0.1 * model.dU([x, y]), c = quivercolor, alpha = quiveralpha,
    #     markershape = :none, arrowhead = :none)

    xlims!(-plotband, plotband)
    ylims!(-plotband, plotband)

    return (plt)
end;

contourplot(model; l=5.0) = plot_pdf(model, Plot_params(rev=:false, plotband=l))

path2pts(path) = map(x -> tuple(x.q...), path)