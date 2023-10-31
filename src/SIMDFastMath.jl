module SIMDFastMath

# Vectorized mathematical functions

# This module exports nothing, its purpose is to specialize
# mathematical functions in Base and Base.FastMath for SIMD.Vec arguments
# using vectorized implementations from SLEEFPirates

import SLEEFPirates as SP
import Base.FastMath as FM
import VectorizationBase as VB
import SIMD
const Vec{T,N} = SIMD.Vec{N,T}

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

# some operators have a fast version in FastMath, but not all
# and some operators have a fast version in SP, but not all !
const not_unops = (:eval, :include, :evalpoly, :hypot, :ldexp, :sincos, :sincos_fast, :pow_fast)
const broken_unops = (:cospi, :sinpi)
unop(n) = !occursin("#", string(n)) && !in(n, union(not_unops, broken_unops))

const unops_SP = filter(unop, names(SP; all = true))
const unops_FM = filter(unop, names(FM; all = true))

# "slow" operators provided by SP
const unops_Base_SP = intersect(unops_SP, names(Base))
# FastMath operators provided by SP
const unops_FM_SP = intersect(unops_SP, unops_FM)
# FastMath operators with only a slow version provided by SP
const unops_FM_SP_slow = filter(unops_SP) do op
    n = Symbol(op, :_fast)
    in(n, unops_FM) && !in(n, unops_SP)
end

# op(x::Vec) = vmap(op,x) = ...
for (mod, unops, fastop) in (
    (Base, unops_Base_SP, identity),
    (FM, unops_FM_SP, identity),
    (FM, unops_FM_SP_slow, sym->Symbol(sym, :_fast)))    

    for op in unops
        op_fast = fastop(op)
        op_SP = getfield(SP, op)
        @eval begin
            @inline $mod.$op_fast(x::Vec{Float32}) = vmap($mod.$op_fast, x) 
            @inline $mod.$op_fast(x::Vec{Float64}) = vmap($mod.$op_fast, x) 
            @inline vmap(::typeof($mod.$op_fast), x) = SIMDVec($op_SP(VBVec(x))) 
        end
    end
end

for op in union(unops_FM_SP, unops_FM_SP_slow), F in (Float32, Float64), N in (4, 8, 16)
    op_fast = getfield(FM, op)
    precompile(op_fast, (Vec{F,N},))
end

# The default implementation of Vec^Int is broken
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{4}) =
let x2 = x * x
        x2 * x2
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{5}) =
    let x2 = x * x
        x2 * x2 * x
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{6}) =
    let x2 = x * x
        x2 * x2 * x2
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{7}) =
    let x2 = x * x
        x2 * x2 * x2 * x
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{8}) =
    let x2 = x * x, x4 = x2 * x2
        x4 * x4
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{9}) =
    let x2 = x * x, x4 = x2 * x2
        x4 * x4 * x
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{10}) =
    let x2 = x * x, x4 = x2 * x2
        x4 * x4 * x2
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{11}) =
    let x2 = x * x, x4 = x2 * x2
        x4 * x4 * x2 * x
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{12}) =
    let x2 = x * x, x4 = x2 * x2
        x4 * x4 * x4
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{13}) =
    let x2 = x * x, x4 = x2 * x2
        x4 * x4 * x4 * x
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{14}) =
    let x2 = x * x, x4 = x2 * x2
        x4 * x4 * x4 * x2
    end
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{15}) =
    let x2 = x * x, x5 = x2 * x2 * x
        x5 * x5 * x5
    end

end
