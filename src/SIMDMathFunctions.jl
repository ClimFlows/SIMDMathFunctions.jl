"""
Vectorized mathematical functions

This module exports nothing, its purpose is to specialize
mathematical functions in Base and Base.FastMath for SIMD.Vec arguments
using vectorized implementations from SLEEFPirates.

See: `is_supported`, `is_fast`, `fast_functions`, `vmap`, `tolerance`.
"""
module SIMDMathFunctions

import SLEEFPirates as SP
import Base.FastMath as FM
import VectorizationBase as VB
import SIMD
const Floats = Union{Float32, Float64}
const Vec{T,N} = SIMD.Vec{N,T} # NB: swapped type parameters
const Vec32{N} = SIMD.Vec{N, Float32}
const Vec64{N} = SIMD.Vec{N, Float64}

"""
    tol = tolerance(fun)
Let `x::SIMD.Vec{N,T}` and `ref` be obtained by applying
`fun` on each element of `x`. Now `fun(x)` may differ
from `ref` by an amount of `tol(fun)*eps(T)*abs(res)`.
`tol==1` except for a few functions, for which `tol==2`.
"""
tolerance(op) = 1

"""

`vmap(fun, x)` applies `fun` to each element of `x::SIMD.Vec` and returns
a `SIMD.Vec`.

    a = vmap(fun, x)

If `fun` returns a 2-uple, `vmap` returns a 2-uple of `SIMD.Vec` :

    a, b = vmap(fun, x)    # `fun(x)` returns a 2-uple, e.g. `sincos`

`vmap(fun, x, y)` works similarly when `fun` takes two input arguments (e.g. `atan(x,y)`)

    a    = vmap(fun, x, y)
    a, b = vmap(fun, x, y) # `fun(x,y)` returns a 2-uple

Generic implementations are provided, which call `fun` and provide no performance
benefit. `vmap` may be specialized for argument `fun`. Such optimized implementations
may return a different result than `fun`, within some tolerance bounds (see [`tolerance`](@ref)).
Currently optimized implementations are provided by `SLEEFPirates.jl`.
"""
@inline vmap(op, x) = vmap_unop(op, x)
@inline vmap(op, x,y) = vmap_binop(op, x, y)

# fallback implementations : calls op on each element of vector
@inline vmap_unop(op, x::Vec) = vec(map(op, values(x)))
@inline vmap_binop(op, x::V, y::V) where {V<:Vec} = vec(map(op, values(x), values(y)))
@inline vmap_binop(op, x::Vec{T}, y::T) where T = vec(map(xx->op(xx,y), values(x)))
@inline vmap_binop(op, x::T, y::Vec{T}) where T = vec(map(yy->op(x,yy), values(y)))
@inline values(x)=map(d->d.value, x.data)
@inline vec(t::NTuple{N, <:SIMD.VecTypes}) where N = SIMD.Vec(t...)
@inline vec(t::NTuple{N, T}) where {N, VT<:SIMD.VecTypes, T<:Tuple{Vararg{VT}}} =
    map(x->SIMD.Vec(x...), tuple(zip(t...)...))

"""
    funs = fast_functions()
    unary_ops = fast_functions(1)
    binary_ops = fast_functions(2)
Returns a vector of fast mathematical functions taking `inputs` input arguments.
"""
fast_functions() =
    [m.sig.parameters[2].instance for m in methods(vmap) if (m.sig.parameters[2]!=Any)]
fast_functions(inputs::Int) =
    [m.sig.parameters[2].instance for m in methods(vmap) if (m.sig.parameters[2]!=Any && length(m.sig.parameters)==inputs+2)]

"""
    flag = is_supported(fun)
Returns `true` if `fun` accepts `SIMD.Vec` arguments.
"""
@inline function is_supported(::F) where {F<:Function}
    V = SIMD.Vec{4,Float64}
    hasmethod(F.instance, Tuple{V}) || hasmethod(F.instance, Tuple{V,V})
end

"""
    flag = is_fast(fun)
Returns `true` if there is a specialization of `vmap` for `fun`,  `false` otherwise.
"""
@inline function is_fast(f::F) where {F<:Function}
    V = SIMD.Vec{4,Float64}
    any(m.sig.parameters[2]==F for m in methods(vmap, Tuple{F, V})) && return true
    any(m.sig.parameters[2]==F for m in methods(vmap, Tuple{F, V, V}))
end

#================ Fast functions from SLEEFPirates =================#

@fastmath begin
    tolerance(::typeof(exp))=2
    tolerance(::typeof(exp10))=2
    tolerance(::typeof(log))=2
    tolerance(::typeof(tanh))=2
    tolerance(::typeof(log10))=2
    tolerance(::typeof(asin))=2
    tolerance(::typeof(^))=2
end
tolerance(::typeof(exp))=2
tolerance(::typeof(exp10))=2
tolerance(::typeof(^))=2
tolerance(::typeof(hypot))=2

# Since SLEEFPirates works with VB.Vec but not with SIMD.Vec,
# we convert between SIMD.Vec and VB.Vec.
# However constructing a VB.Vec of length exceeding the native vector length
# returns a VB.VecUnroll => we must handle also this type

# Constructors SIMD.Vec and VB.Vec accept x... as arguments where x is iterable
# so we make SIMD.Vec and VB.VecUnroll iterable (VB.Vec can be converted to Tuple).
# To avoid messing up existing behavior of Base.iterate for SIMD and VB types, we define a wrapper type Iter{V}

struct Iter{V}
    vec::V
end
@inline Base.iterate(v::Iter, args...) = iter(v.vec, args...)

# iterate over SIMD.Vec
@inline iter(v::SIMD.Vec) = v[1], 2
@inline iter(v::SIMD.Vec{N}, i) where {N} = (i > N ? nothing : (v[i], i + 1))

# iterate over VB.VecUnroll
@inline function iter(v::VB.VecUnroll)
    data = VB.data(v)
    return data[1](1), (1, 1)
end
@inline function iter(v::VB.VecUnroll{N,W}, (i, j)) where {N,W}
    data = VB.data(v)
    if j < W
        return data[i](j + 1), (i, j + 1)
    elseif i <= N # there are N+1 vectors
        return data[i+1](1), (i + 1, 1)
    else
        return nothing
    end
end

@inline SIMDVec(v::VB.Vec) = SIMD.Vec(Tuple(v)...)
@inline SIMDVec(vu::VB.VecUnroll) = SIMD.Vec(Iter(vu)...)
@inline VBVec(v::Vec) = VB.Vec(Iter(v)...)
@inline VBVec(v::Floats) = v

# some operators have a fast version in FastMath, but not all
# and some operators have a fast version in SP, but not all !
const not_unops = (:eval, :include, :evalpoly, :hypot, :ldexp, :sincos, :sincos_fast, :pow_fast)
const broken_unops = (:cospi, :sinpi)
is_unop(n) = !occursin("#", string(n)) && !in(n, union(not_unops, broken_unops))

const unops_SP = filter(is_unop, names(SP; all = true))
const unops_FM = filter(is_unop, names(FM; all = true))

# "slow" operators provided by SP
const unops_Base_SP = intersect(unops_SP, names(Base))
# FastMath operators provided by SP
const unops_FM_SP = intersect(unops_SP, unops_FM)
# FastMath operators with only a slow version provided by SP
const unops_FM_SP_slow = filter(unops_SP) do op
    n = Symbol(op, :_fast)
    in(n, unops_FM) && !in(n, unops_SP)
end

# one input, one output
for (mod, unops, fastop) in (
    (Base, unops_Base_SP, identity),
    (FM, unops_FM_SP, identity),
    (FM, unops_FM_SP_slow, sym->Symbol(sym, :_fast)))

    for op in unops
        op_fast = fastop(op)
        op_SP = getfield(SP, op)
        @eval begin
            @inline $mod.$op_fast(x::Vec32) = vmap($mod.$op_fast, x)
            @inline $mod.$op_fast(x::Vec64) = vmap($mod.$op_fast, x)
            @inline vmap(::typeof($mod.$op_fast), x) = SIMDVec($op_SP(VBVec(x)))
        end
    end
end

# one input, two outputs
for (mod, op) in ((Base, :sincos), (FM, :sincos_fast))
    @eval begin
        @inline $mod.$op(x::Vec{<:Floats}) = vmap($mod.$op, x)
        @inline vmap(::typeof($mod.$op), x) = map(SIMDVec, SP.$op(VBVec(x)))
    end
end

# two inputs, one output
binops = ((Base,:hypot,SP.hypot), (Base,:^,SP.pow), (FM,:pow_fast, SP.pow_fast))
for (mod, op_slow, op_fast) in binops
    @eval begin
        @inline $mod.$op_slow(x::Vec{T, N}, y::Vec{T, N}) where {T<:Floats, N} = vmap($mod.$op_slow, x,y)
        @inline $mod.$op_slow(x::T, y::Vec{T}) where {T<:Floats} = vmap($mod.$op_slow, x,y)
        @inline $mod.$op_slow(x::Vec{T}, y::T) where {T<:Floats} = vmap($mod.$op_slow, x,y)
        @inline vmap(::typeof($mod.$op_slow), x, y) = SIMDVec($op_fast(VBVec(x), VBVec(y)))
    end
end

# precompilation
for op in fast_functions(1), F in (Float32, Float64), N in (4, 8, 16)
    precompile(op, (Vec{F,N},))
end

for op in fast_functions(2), F in (Float32, Float64), N in (4, 8, 16)
    precompile(op, (Vec{F,N},Vec{F,N}))
    precompile(op, (Vec{F,N},F))
    precompile(op, (F,Vec{F,N}))
end

end
