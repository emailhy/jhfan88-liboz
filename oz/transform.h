//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
#pragma once

#include <oz/gpu_plm2.h>

namespace oz {

    template<typename Arg, typename Result> struct unary_function {
        typedef Arg argument_type;
        typedef Result result_type;
    };

    template<typename Arg1, typename Arg2, typename Result> struct binary_function {
        typedef Arg1 first_argument_type;
        typedef Arg2 second_argument_type;
        typedef Result result_type;
    };

    template<typename Arg1, typename Arg2, typename Arg3, typename Result> struct ternary_function {
        typedef Arg1 first_argument_type;
        typedef Arg2 second_argument_type;
        typedef Arg3 third_argument_type;
        typedef Result result_type;
    };

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Result> struct quaternary_function {
        typedef Arg1 first_argument_type;
        typedef Arg2 second_argument_type;
        typedef Arg3 third_argument_type;
        typedef Arg4 forth_argument_type;
        typedef Result result_type;
    };

    namespace detail {
        template<typename Arg, typename Result, typename F>
        __global__ void transform_global( const gpu_plm2<Arg> src, gpu_plm2<Result> dst, const F f ) {
            int ix = blockDim.x * blockIdx.x + threadIdx.x;
            int iy = blockDim.y * blockIdx.y + threadIdx.y;
            if(ix >= dst.w || iy >= dst.h)
                return;
            dst.write(ix, iy, f(src(ix, iy)));
        }

        template<typename Arg1, typename Arg2, typename Result, typename F>
        __global__ void transform_global( const gpu_plm2<Arg1> src1, const gpu_plm2<Arg2> src2,
                                          gpu_plm2<Result> dst, const F f )
        {
            int ix = blockDim.x * blockIdx.x + threadIdx.x;
            int iy = blockDim.y * blockIdx.y + threadIdx.y;
            if(ix >= dst.w || iy >= dst.h)
                return;
            dst.write(ix, iy, f(src1(ix, iy), src2(ix, iy)));
        }

        template<typename Arg1, typename Arg2, typename Arg3, typename Result, typename F>
        __global__ void transform_global( const gpu_plm2<Arg1> src1, const gpu_plm2<Arg2> src2, const gpu_plm2<Arg3> src3,
                                          gpu_plm2<Result> dst, const F f )
        {
            int ix = blockDim.x * blockIdx.x + threadIdx.x;
            int iy = blockDim.y * blockIdx.y + threadIdx.y;
            if(ix >= dst.w || iy >= dst.h)
                return;
            dst.write(ix, iy, f(src1(ix, iy), src2(ix, iy), src3(ix, iy)));
        }

        template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Result, typename F>
        __global__ void transform_global( const gpu_plm2<Arg1> src1, const gpu_plm2<Arg2> src2, const gpu_plm2<Arg3> src3,
                                          const gpu_plm2<Arg4> src4, gpu_plm2<Result> dst, const F f )
        {
            int ix = blockDim.x * blockIdx.x + threadIdx.x;
            int iy = blockDim.y * blockIdx.y + threadIdx.y;
            if(ix >= dst.w || iy >= dst.h)
                return;
            dst.write(ix, iy, f(src1(ix, iy), src2(ix, iy), src3(ix, iy), src4(ix, iy)));
        }
    }

    template<typename Arg, typename Result, typename F>
    void transform_unary( const gpu_image& src, gpu_image& dst, const F& f ) {
        OZ_CHECK_FORMAT(src.format(), type_traits<Arg>::format());
        OZ_CHECK_FORMAT(dst.format(), type_traits<Result>::format());
        dim3 threads(8, 8);
        dim3 blocks((dst.w()+threads.x-1)/threads.x, (dst.h()+threads.y-1)/threads.y);
        detail::transform_global<<<blocks, threads>>>(gpu_plm2<Arg>(src), gpu_plm2<Result>(dst), f);
        OZ_CUDA_ERROR_CHECK();
    }

    template<typename Arg, typename Result, typename F>
    gpu_image transform_unary( const gpu_image& src, const F& f ) {
        gpu_image dst(src.size(), type_traits<Result>::format());
        transform_unary<Arg,Result,F>(src, dst, f);
        return dst;
    }

    template<typename F>
    gpu_image transform( const gpu_image& src, const F& f ) {
        typedef typename F::argument_type Arg;
        typedef typename F::result_type Result;
        return transform_unary<Arg,Result,F>(src, f);
    }

//     template<typename F>
//     gpu_image transform_f( const gpu_image& src, const F& f) {
//         switch (src.format()) {
//             case FMT_FLOAT:  return transform_unary<float, float >(src, f);
//             case FMT_FLOAT2: return transform_unary<float2,float2>(src, f);
//             case FMT_FLOAT3: return transform_unary<float3,float3>(src, f);
//             case FMT_FLOAT4: return transform_unary<float4,float4>(src, f);
//             default:
//                 OZ_INVALID_FORMAT();
//         }
//     }


    template<typename Arg1, typename Arg2, typename Result, typename F>
    void transform_binary( const gpu_image& src1, const gpu_image& src2, gpu_image& dst, const F& f ) {
        if ((src1.size() != src2.size()) || (src1.size() != dst.size())) OZ_INVALID_SIZE();
        OZ_CHECK_FORMAT(src1.format(), type_traits<Arg1>::format());
        OZ_CHECK_FORMAT(src2.format(), type_traits<Arg2>::format());
        OZ_CHECK_FORMAT(dst.format(), type_traits<Result>::format());
        dim3 threads(8, 8);
        dim3 blocks((dst.w()+threads.x-1)/threads.x, (dst.h()+threads.y-1)/threads.y);
        detail::transform_global<<<blocks, threads>>>(gpu_plm2<Arg1>(src1), gpu_plm2<Arg2>(src2), gpu_plm2<Result>(dst), f);
        OZ_CUDA_ERROR_CHECK();
    }

    template<typename Arg1, typename Arg2, typename Result, typename F>
    gpu_image transform_binary( const gpu_image& src1, const gpu_image& src2, const F& f ) {
        gpu_image dst(src1.size(), type_traits<Result>::format());
        transform_binary<Arg1,Arg2,Result,F>(src1, src2, dst, f);
        return dst;
    }

    template<typename F>
    gpu_image transform( const gpu_image& src1, const gpu_image& src2, const F& f ) {
        typedef typename F::first_argument_type Arg1;
        typedef typename F::second_argument_type Arg2;
        typedef typename F::result_type Result;
        return transform_binary<Arg1,Arg2,Result,F>(src1, src2, f);
    }


    template<typename Arg1, typename Arg2, typename Arg3, typename Result, typename F>
    void transform_ternary( const gpu_image& src1, const gpu_image& src2, const gpu_image& src3,
                            gpu_image& dst, const F& f )
    {
        if ((src1.size() != src2.size()) ||
            (src1.size() != src3.size()) ||
            (src1.size() != dst.size())) OZ_INVALID_SIZE();
        OZ_CHECK_FORMAT(src1.format(), type_traits<Arg1>::format());
        OZ_CHECK_FORMAT(src2.format(), type_traits<Arg2>::format());
        OZ_CHECK_FORMAT(src3.format(), type_traits<Arg3>::format());
        OZ_CHECK_FORMAT(dst.format(), type_traits<Result>::format());
        dim3 threads(8, 8);
        dim3 blocks((dst.w()+threads.x-1)/threads.x, (dst.h()+threads.y-1)/threads.y);
        detail::transform_global<<<blocks, threads>>>(
            gpu_plm2<Arg1>(src1), gpu_plm2<Arg2>(src2), gpu_plm2<Arg3>(src3), gpu_plm2<Result>(dst), f);
        OZ_CUDA_ERROR_CHECK();
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Result, typename F>
    gpu_image transform_ternary( const gpu_image& src1, const gpu_image& src2,
                                const gpu_image& src3, const F& f )
    {
        gpu_image dst(src1.size(), type_traits<Result>::format());
        transform_ternary<Arg1,Arg2,Arg3,Result,F>(src1, src2, src3, dst, f);
        return dst;
    }

    template<typename F>
    gpu_image transform( const gpu_image& src1, const gpu_image& src2,
                        const gpu_image& src3, const F& f )
    {
        typedef typename F::first_argument_type Arg1;
        typedef typename F::second_argument_type Arg2;
        typedef typename F::third_argument_type Arg3;
        typedef typename F::result_type Result;
        return transform_ternary<Arg1,Arg2,Arg3,Result,F>(src1, src2, src3, f);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Result, typename F>
    void transform_quaternary( const gpu_image& src1, const gpu_image& src2, const gpu_image& src3,
                               const gpu_image& src4, gpu_image& dst, const F& f )
    {
        if ((src1.size() != src2.size()) ||
            (src1.size() != src3.size()) ||
            (src1.size() != src4.size()) ||
            (src1.size() != dst.size())) OZ_INVALID_SIZE();
        OZ_CHECK_FORMAT(src1.format(), type_traits<Arg1>::format());
        OZ_CHECK_FORMAT(src2.format(), type_traits<Arg2>::format());
        OZ_CHECK_FORMAT(src3.format(), type_traits<Arg3>::format());
        OZ_CHECK_FORMAT(src4.format(), type_traits<Arg4>::format());
        OZ_CHECK_FORMAT(dst.format(), type_traits<Result>::format());
        dim3 threads(8, 8);
        dim3 blocks((dst.w()+threads.x-1)/threads.x, (dst.h()+threads.y-1)/threads.y);
        detail::transform_global<<<blocks, threads>>>(
            gpu_plm2<Arg1>(src1), gpu_plm2<Arg2>(src2), gpu_plm2<Arg3>(src3), gpu_plm2<Arg4>(src4), gpu_plm2<Result>(dst), f);
        OZ_CUDA_ERROR_CHECK();
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Result, typename F>
    gpu_image transform_quaternary( const gpu_image& src1, const gpu_image& src2, const gpu_image& src3,
                                    const gpu_image& src4, const F& f )
    {
        gpu_image dst(src1.size(), type_traits<Result>::format());
        transform_quaternary<Arg1,Arg2,Arg3,Arg4,Result,F>(src1, src2, src3, src4,dst, f);
        return dst;
    }

    template<typename F>
    gpu_image transform( const gpu_image& src1, const gpu_image& src2, const gpu_image& src3,
                         const gpu_image& src4, const F& f )
    {
        typedef typename F::first_argument_type Arg1;
        typedef typename F::second_argument_type Arg2;
        typedef typename F::third_argument_type Arg3;
        typedef typename F::forth_argument_type Arg4;
        typedef typename F::result_type Result;
        return transform_quaternary<Arg1,Arg2,Arg3,Arg4,Result,F>(src1, src2, src3, src4, f);
    }


    template<typename UnaryFunction>
    gpu_image transform_f( const gpu_image& src, UnaryFunction f) {
        switch (src.format()) {
            case FMT_FLOAT:  return transform<float, float >(src, f);
            case FMT_FLOAT2: return transform<float2,float2>(src, f);
            case FMT_FLOAT3: return transform<float3,float3>(src, f);
            case FMT_FLOAT4: return transform<float4,float4>(src, f);
            default:
                OZ_INVALID_FORMAT();
        }
    }

    template<typename BinaryFunction>
    gpu_image transform_f( const gpu_image& src0, const gpu_image& src1, BinaryFunction f) {
        switch (src0.format()) {
            case FMT_FLOAT:  return transform<float, float, float >(src0, src1, f);
            case FMT_FLOAT2: return transform<float2,float2,float2>(src0, src1, f);
            case FMT_FLOAT3: return transform<float3,float3,float3>(src0, src1, f);
            case FMT_FLOAT4: return transform<float4,float4,float4>(src0, src1, f);
            default:
                OZ_INVALID_FORMAT();
        }
    }

    template<typename BinaryFunction>
    gpu_image transform_f( const gpu_image& src0, const gpu_image& src1,
                          const gpu_image& src2, BinaryFunction f)
    {
        switch (src0.format()) {
            case FMT_FLOAT:  return transform<float, float, float, float >(src0, src1, src2, f);
            case FMT_FLOAT2: return transform<float2,float2,float2,float2>(src0, src1, src2, f);
            case FMT_FLOAT3: return transform<float3,float3,float3,float3>(src0, src1, src2, f);
            case FMT_FLOAT4: return transform<float4,float4,float4,float4>(src0, src1, src2, f);
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
