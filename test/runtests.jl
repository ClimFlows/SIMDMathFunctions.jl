using SIMDFastMath:
    SIMD, tolerance, fast_functions, is_supported, is_fast, vmap_unop, vmap_binop
using Test

@info "Fast functions"
for fun in sort(map(string, fast_functions()))
    @info "--- $fun"
end

data(F, N, ::Function) = range(F(0.01), F(0.9), length = N)
data(F, N, ::typeof(acosh)) = range(F(1.1), F(1.9), length = N)
data(F, N, ::typeof(@fastmath acosh)) = range(F(1.1), F(1.9), length = N)

data_binop(F, N, ::Function) =
    range(F(0.01), F(0.9), length = N), range(F(0.01), F(0.9), length = N)

function validate(res::SIMD.Vec, ref, tol)
    err = relative_error(res, ref)
    err, any(err > tol)
end
function validate(res::Tuple, ref, tol)
    err = map(relative_error, res, ref)
    err, any(map(err -> any(err > tol), err))
end
relative_error(res, ref) = abs(res - ref) / abs(ref)

for fun in sort(fast_functions(2), by = string) # two-argument functions
    @assert is_supported(fun)
    @assert is_fast(fun)
    tol = tolerance(fun)
    tol = tolerance(fun)
    @testset "$(string(fun))" begin
        for F in (Float32, Float64), N in (4, 8, 16, 32)
            x, y = data_binop(F, N, fun)
            x, y = SIMD.Vec(x...), SIMD.Vec(x...)
            res, ref = fun(x, y), vmap_binop(fun, x, y)
            err, fail = validate(res, ref, tol * eps(F))
            fail && @warn fun arg ref res err
            @test !fail
        end
    end
end

for fun in sort(fast_functions(1), by = string) # one-argument functions
    @assert is_supported(fun)
    @assert is_fast(fun)
    tol = tolerance(fun)
    @testset "$(string(fun))" begin
        for F in (Float32, Float64), N in (4, 8, 16, 32)
            d = SIMD.Vec(data(F, N, fun)...)
            res, ref = fun(d), vmap_unop(fun, d)
            err, fail = validate(res, ref, tol * eps(F))
            fail && @warn fun arg ref res err
            @test !fail
        end
    end
end
