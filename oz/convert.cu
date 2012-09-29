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
#include <oz/convert.h>
#include <oz/gpu_plm2.h>
#include <oz/transform.h>
#include <oz/convert_type.h>


namespace oz {
    template<typename Arg, typename Result>
    struct op_convert : public unary_function<Arg,Result> {
        inline __host__ __device__ Result operator()( Arg s ) const {
            return convert_type<Arg,Result>(s);
        }
    };


    template<typename Arg, typename Result, typename I>
    static I convert2( const I& src ) {
        if (!src.is_valid()) return gpu_image();
        return transform(src, op_convert<Arg,Result>());
    }


    template<typename Result, typename I>
    static I convert1( const I& src ) {
        switch (src.format()) {
            case FMT_UCHAR:  return convert2<uchar, Result>(src);
            case FMT_UCHAR2: return convert2<uchar2,Result>(src);
            case FMT_UCHAR3: return convert2<uchar3,Result>(src);
            case FMT_UCHAR4: return convert2<uchar4,Result>(src);
            case FMT_FLOAT:  return convert2<float, Result>(src);
            case FMT_FLOAT2: return convert2<float2,Result>(src);
            case FMT_FLOAT3: return convert2<float3,Result>(src);
            case FMT_FLOAT4: return convert2<float4,Result>(src);
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename I>
    static I convert0( const I& src, image_format_t format, bool clone ) {
        switch (format) {
            case FMT_UCHAR:  return convert1<uchar >(src);
            case FMT_UCHAR2: return convert1<uchar2>(src);
            case FMT_UCHAR3: return convert1<uchar3>(src);
            case FMT_UCHAR4: return convert1<uchar4>(src);
            case FMT_FLOAT:  return convert1<float >(src);
            case FMT_FLOAT2: return convert1<float2>(src);
            case FMT_FLOAT3: return convert1<float3>(src);
            case FMT_FLOAT4: return convert1<float4>(src);
            default:
                OZ_INVALID_FORMAT();
        }
    }


    cpu_image convert( const cpu_image& src, image_format_t format, bool clone ) {
        return convert0(src, format, clone);
    }


    gpu_image convert( const gpu_image& src, image_format_t format, bool clone ) {
        return convert0(src, format, clone);
    }


    template<typename I>
    static I to_uchar_I( const I& src, bool clone ) {
        switch (src.format()) {
            case FMT_UCHAR:
            case FMT_UCHAR2:
            case FMT_UCHAR3:
            case FMT_UCHAR4:
                return clone? src.clone() : src;

            case FMT_FLOAT:  return convert0(src, FMT_UCHAR,  clone);
            case FMT_FLOAT2: return convert0(src, FMT_UCHAR2, clone);
            case FMT_FLOAT3: return convert0(src, FMT_UCHAR3, clone);
            case FMT_FLOAT4: return convert0(src, FMT_UCHAR4, clone);

            default:
                OZ_INVALID_FORMAT();
        }
    }


    cpu_image to_uchar( const cpu_image& src, bool clone ) {
        return to_uchar_I(src, clone);
    }


    gpu_image to_uchar( const gpu_image& src, bool clone ) {
        return to_uchar_I(src, clone);
    }


    template<typename I>
    static I to_float_I( const I& src, bool clone ) {
        switch (src.format()) {
            case FMT_FLOAT:
            case FMT_FLOAT2:
            case FMT_FLOAT3:
            case FMT_FLOAT4:
                return clone? src.clone() : src;

            case FMT_UCHAR:  return convert0(src, FMT_FLOAT,  clone);
            case FMT_UCHAR2: return convert0(src, FMT_FLOAT2, clone);
            case FMT_UCHAR3: return convert0(src, FMT_FLOAT3, clone);
            case FMT_UCHAR4: return convert0(src, FMT_FLOAT4, clone);

            default:
                OZ_INVALID_FORMAT();
        }
    }


    cpu_image to_float( const cpu_image& src, bool clone ) {
        return to_float_I(src, clone);
    }


    gpu_image to_float( const gpu_image& src, bool clone ) {
        return to_float_I(src, clone);
    }
}