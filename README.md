# SIMDMathFunctions

Fast vectorized mathematical functions for SIMD.jl , using SLEEFPirates.jl .

[![CI](https://github.com/ClimFlows/SIMDMathFunctions.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ClimFlows/SIMDMathFunctions.jl/actions/workflows/CI.yml)
[![Code Coverage](https://codecov.io/gh/ClimFlows/SIMDMathFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/SIMDMathFunctions)

## Installing

This package is registered. To install it :
```Julia
] add SIMDMathFunctions
```

## Overview

The primary goal of `SIMDMathFunctions` is to provide efficient methods for mathematical functions with `SIMD.Vec` arguments. Under the hood, optimized implementations provided by `SLEEFPirates.jl` are used. This allows explicitly vectorized code using `SIMD.jl` to benefit from fast vectorized math functions.

```Julia
using SIMD: VecRange
using SIMDMathFunctions: is_supported, is_fast, fast_functions
using BenchmarkTools

function exp!(xs::Vector{T}, ys::Vector{T}) where {T}
    @inbounds for i in eachindex(xs,ys)
        xs[i] = @fastmath exp(ys[i])
    end
end

function exp!(xs::Vector{T}, ys::Vector{T}, ::Val{N}) where {N, T}
    @assert length(ys) == length(xs)
    @assert length(xs) % N == 0
    @assert is_supported(@fastmath exp)
    @inbounds for istart in 1:N:length(xs)
        i = VecRange{N}(istart)
        xs[i] = @fastmath exp(ys[i])
    end
end

y=randn(Float32, 1024*1024); x=similar(y);

@benchmark exp!($x, $y)
@benchmark exp!($x, $y, Val(8))
@benchmark exp!($x, $y, Val(16))
@benchmark exp!($x, $y, Val(32))

is_fast(exp)
unary_funs = fast_functions(1)
binary_funs = fast_functions(2)
```

`is_supported(fun)` returns `true` if function `fun` supports `SIMD.Vec` arguments. Similarly `is_fast(fun)` returns `true` if `fun` has an optimized implementation.

`fast_functions([ninputs])` returns a vector of functions benefitting from a fast implementation, restricted to those accepting `ninputs` input arguments if `ninputs` is provided.

`SIMDMathFunctions` also provides a helper function `vmap` to vectorize not-yet-supported mathematical functions. For example :

```Julia
using SIMD: Vec
import SIMDMathFunctions: vmap
import SpecialFunctions: erf

erf(x::Vec) = vmap(erf, x)
erf(x::Vec, y::Vec) = vmap(erf, x, y)
erf(x::Vec{N,T}, y::T) where {N,T} = vmap(erf, x, y)

x = Vec(randn(Float32, 16)...)
@benchmark erf($x)
```

The default `vmap` method simply calls `erf` on each element of `x`. There is no performance benefit, but it allows generic code to use `erf`. If `erf_SIMD` is optimized for vector inputs, you can provide a specialized method for `vmap`:

```Julia
using VectorizationBase: verf # vectorized implementation
using SIMDMathFunctions: SIMDVec, VBVec # VectorizationBase <=> SIMD conversion

erf_SIMD(x) = SIMDVec(verf(VBVec(x)))
vmap(::typeof(erf), x) = erf_SIMD(x)

@benchmark erf($x)
```
