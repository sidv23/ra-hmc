import Base: +, -

abstract type AbstractSampler end

abstract type State end

@with_kw mutable struct Model
    ξ
    d = length(ξ)
    f = x -> max(pdf(ξ, x), floatmin(Float64))
    g = x -> Zygote.gradient(x_ -> Zygote.forwarddiff(f, x_), x)
    U = x -> min(-logpdf(ξ, x), floatmax(Float64))
    dU = x -> Zygote.gradient(x_ -> Zygote.forwarddiff(U, x_), x)[1]
    # dU = x -> -DistributionsAD.gradlogpdf(ξ, x)
end;