begin
    m2t = x -> tuple.(eachcol(x)...)                           # Convert from Matrix -> Tuples
    m2a = x -> collect(eachrow(x))                             # Convert from Matrix -> Arrays
    t2m = T -> hcat(collect.(T)...)'                           # Convert from Tuples -> Matrix
    t2a = T -> [[a...] for a in T]                             # Convert from Tuples -> Arrays
    a2t = V -> Tuple.(V)                                       # Convert from Arrays -> Tuples
    a2m = V -> Matrix(reduce(hcat, V)')                                # Convert from Arrays -> Matrix
end

function -(s1::NamedTuple, s2::NamedTuple)
    p = s1.p .- s2.p
    q = s1.q .- s2.q
    return (; s1..., p = p, q = q)
end

function +(s1::NamedTuple, s2::NamedTuple)
    p = s1.p .+ s2.p
    q = s1.q .+ s2.q
    return (; s1..., p = p, q = q)
end

function KE(state::NamedTuple, M::Model)
    sum((state.p .^ 2) ./ (2 .* state.m))
end

function PE(state::NamedTuple, M::Model)
    M.U(state.q)
end

function H(state::NamedTuple, M::Model)
    PE(state, M) + KE(state, M)
end

