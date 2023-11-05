using SIMDFastMath: SIMD, tolerance, fast_functions, is_supported, is_fast
using Test

@info "Fast functions"
for fun in sort(map(string, fast_functions()))
    @info "--- $fun"
end

data(F, N, ::Function) = range(F(0.01), F(0.9), length = N)
data(F, N, ::typeof(acosh)) = range(F(1.1), F(1.9), length = N)
data(F, N, ::typeof(@fastmath acosh)) = range(F(1.1), F(1.9), length = N)

function validate(res::SIMD.Vec, ref, tol)
    ref = SIMD.Vec(ref...)
    err = relative_error(res, ref)
    ref, err, any( err > tol)
end
function validate(res::Tuple, ref, tol)
    ref = map(x->SIMD.Vec(x...), Tuple(zip(ref...)))
    err = map( relative_error, res,ref )
    ref, err, any( map( err -> any(err>tol), err) )
end
relative_error(res, ref) = abs(res - ref)/abs(ref)

for fun in sort(fast_functions(1), by=string) # one-argument functions
    @assert is_supported(fun)
    @assert is_fast(fun)
    tol = tolerance(fun)
    @testset "$(string(fun))" begin
        for F in (Float32, Float64), N in (4, 8, 16, 32)
            d = data(F, N, fun)
            res = fun(SIMD.Vec(d...))
            ref, err, fail = validate(res, map(fun, d), tol*eps(F))
            if fail
                @warn fun arg ref res err
                @test false
            else
                @test true
            end
        end
    end
end
