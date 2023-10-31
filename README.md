# SIMDFastMath

Fast vectorized mathematical functions for SIMD.jl , using SLEEFPirates.jl .

[![CI](https://github.com/dubosipsl/SIMDFastMath/actions/workflows/CI.yml/badge.svg)](https://github.com/dubosipsl/SIMDFastMath/actions/workflows/CI.yml)
[![Code Coverage](https://codecov.io/gh/dubosipsl/SIMDFastMath/branch/main/graph/badge.svg)](https://codecov.io/gh/dubosipsl/SIMDFastMath) 

## Overview

When loaded, this package provides mathematical functions accepting `SIMD.Vec` arguments. Under the hood, optimized implementations provided by `SLEEFPirates.jl` are used. This allows explicitly vectorized code using `SIMD.jl` to benefit from fast vectorized math functions.

```Julia
using SIMD: VecRange
using SIMDFastMath: is_supported
using BenchmarkTools

function exp!(xs::Vector{T}, ys::Vector{T}) where {T}
    @inbounds for i in eachindex(xs,ys)
        xs[i] = @fastmath exp(ys[i])
    end
end

function vexp!(xs::Vector{T}, ys::Vector{T}, ::Val{N}) where {N, T}
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
@benchmark vexp!($x, $y, Val(8))
@benchmark vexp!($x, $y, Val(16))
@benchmark vexp!($x, $y, Val(32))

@btime is_supported(sin)
```

`SIMDFastMath.is_supported(fun)` is a zero-cost function returning `true` if function `fun` is supported. `SIMDFastMath.supported()` returns a vector of supported functions.


## Installing

This package is not yet registered. To install it :
```Julia
] add https://github.com/dubosipsl/SIMDFastMath
```
