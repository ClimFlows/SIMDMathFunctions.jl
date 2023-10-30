using SIMDFastMath: SIMD, unops_Base_SP, unops_FM_SP, unops_FM_SP_slow
using Test

data(F, N, ::Function) = range(F(0.01), F(0.9), length = N)
data(F, N, ::typeof(acosh)) = range(F(1.1), F(1.9), length = N)

tolerance(op) = 1
tolerance(::typeof(@fastmath log))=2
tolerance(::typeof(@fastmath tanh))=2
tolerance(::typeof(exp))=2
tolerance(::typeof(exp10))=2

for (mod, unops) in (
    (Base, unops_Base_SP),
    (Base.FastMath, unops_FM_SP),
    (Base.FastMath, unops_FM_SP_slow))
    for unop in unops
        op = getfield(mod, unop)
        tol = tolerance(op)
        @testset "$(string(mod)).$(string(op))" begin
            for F in (Float32, Float64), N in (4, 8, 16, 32)
                d = data(F, N, op)
                arg = SIMD.Vec(d...)
                ref = SIMD.Vec(map(op, d)...)
                res = op(arg)
                err = abs(res - ref)/abs(ref)
                if any( err > tol*eps(F))
                    @warn op arg ref res err
                    @test false
                else
                    @test true
                end
            end
        end
    end
end
