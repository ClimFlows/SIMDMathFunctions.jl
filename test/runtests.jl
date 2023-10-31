using SIMDFastMath: SIMD, vmap, tolerance, supported #unops_Base_SP, unops_FM_SP, unops_FM_SP_slow
using Test

@info "Supported functions"
for fun in supported()
    @info string(fun)
end

data(F, N, ::Function) = range(F(0.01), F(0.9), length = N)
data(F, N, ::typeof(acosh)) = range(F(1.1), F(1.9), length = N)
data(F, N, ::typeof(@fastmath acosh)) = range(F(1.1), F(1.9), length = N)

for fun in supported()
    tol = tolerance(fun)
    @testset "$(string(fun))" begin
        for F in (Float32, Float64), N in (4, 8, 16, 32)
            d = data(F, N, fun)
            arg = SIMD.Vec(d...)
            ref = SIMD.Vec(map(fun, d)...)
            res = fun(arg)
            err = abs(res - ref)/abs(ref)
            if any( err > tol*eps(F))
                @warn fun arg ref res err
                @test false
            else
                @test true
            end
        end
    end
end
