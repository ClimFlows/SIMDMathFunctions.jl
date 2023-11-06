using SIMDFastMath:
    SIMD, tolerance, fast_functions, is_supported, is_fast, vmap_unop, vmap_binop
using Test

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

@testset "Two-argument functions" begin
    for fun in sort(fast_functions(2), by = string)
        @assert is_supported(fun)
        @assert is_fast(fun)
        @info "--- $(string(fun))"
        tol = tolerance(fun)
        for F in (Float32, Float64), N in (4, 8, 16, 32)
            x, y = data_binop(F, N, fun)
            xv, yv = SIMD.Vec(x...), SIMD.Vec(x...)
            for (xx, yy) in ((xv, yv), (x[N>>1], yv), (xv, y[N>>1]))
                res, ref = fun(xx, yy), vmap_binop(fun, xx, yy)
                err, fail = validate(res, ref, tol * eps(F))
                fail && @warn fun (xx, yy) ref res err
                @test !fail
            end
        end
    end
end

@testset "One-argument functions" begin
    for fun in sort(fast_functions(1), by = string)
        @assert is_supported(fun)
        @assert is_fast(fun)
        @info "--- $(string(fun))"
        tol = tolerance(fun)
        for F in (Float32, Float64), N in (4, 8, 16, 32)
            d = SIMD.Vec(data(F, N, fun)...)
            res, ref = fun(d), vmap_unop(fun, d)
            err, fail = validate(res, ref, tol * eps(F))
            fail && @warn fun arg ref res err
            @test !fail
        end
    end
end
