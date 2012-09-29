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
#include <oz/gpu_image.h>
#include <oz/transform.h>
#include <oz/transform_channel.h>
#include <oz/foreach.h>
#include <oz/generate.h>


namespace oz {

    template<typename T> struct op_neg : public unary_function<T,T> {
        inline __device__ T operator()( T s ) const { return -s; }
    };

    template<typename T> struct op_add : public binary_function<T,T,T> {
        inline __device__ T operator()( T a, T b ) const { return a + b; }
    };

    struct op_scalar_add {
        float k_;
        op_scalar_add( float k ) : k_(k) {}
        inline __device__ float operator()( float s ) const { return s + k_; }
    };

    template<typename T> struct op_sub : public binary_function<T,T,T> {
        inline __device__ T operator()( T a, T b ) const { return a - b; }
    };

    template<typename T1, typename T2> struct op_mul : public binary_function<T1,T2,T1> {
        inline __device__ T1 operator()( T1 a, T2 b ) const { return a * b; }
    };

    template<typename T1, typename T2> struct op_div : public binary_function<T1,T2,T1> {
        inline __device__ T1 operator()( T1 a, T2 b ) const { return a / b; }
    };

    template<typename T> struct op_scalar_mul : public unary_function<T,T> {
        float k_;
        op_scalar_mul( float k ) : k_(k) {}
        inline __device__ T operator()( T s ) const { return s * k_; }
    };

    template<typename T> struct op_vec_mul : public unary_function<T,T> {
        T v_;
        op_vec_mul( T v ) : v_(v) {}
        inline __device__ T operator()( T s ) const { return s * v_; }
    };

    template<typename T> struct op_fill {
        gpu_plm2<T> dst_;
        T value_;
        int2 delta_;
        op_fill( gpu_image& dst, int2 delta, T value ) : dst_(dst), delta_(delta), value_(value) {}
        inline __device__ void operator()( int ix, int iy) {
            dst_.write(delta_.x + ix, delta_.y + iy, value_);
        }
    };

    struct op_adjust {
        float a_, b_;
        op_adjust( float a, float b ) : a_(a), b_(b) {}
        inline __device__ float operator()( float s ) const { return s * a_ + b_; }
    };

    struct op_invert {
        inline __device__ float operator()( float s ) const { return 1-__saturatef(s); }
    };

    struct op_saturate {
        inline __device__ float operator()( float s ) const { return __saturatef(s); }
    };

    struct op_clamp {
        float a_, b_;
        op_clamp( float a, float b ) : a_(a), b_(b) {}
        inline __device__ float operator()( float s ) const { return clamp(s, a_, b_); }
    };

    template<typename T> struct op_vec_clamp : unary_function<T,T> {
        T a_, b_;
        op_vec_clamp( T a, T b ) : a_(a), b_(b) {}
        inline __device__ T operator()( T s ) const { return clamp(s, a_, b_); }
    };

    struct op_lerp {
        float t_;
        op_lerp( float t ) : t_(t) {}
        inline __device__ float operator()( float a, float b ) const { return (1 - t_) * a + t_ * b; }
    };

    template<typename T> struct op_abs : public unary_function<T,float> {
        inline __device__ float operator()( T a ) const { return length(a); }
    };

    template<typename T> struct op_abs2 : public unary_function<T,float> {
        inline __device__ float operator()( T a ) const { return dot(a,a); }
    };

    struct op_sqrt {
        inline __device__ float operator()( float a ) const { return sqrtf(a); }
    };

    struct op_sqr {
        inline __device__ float operator()( float a ) const { return a*a; }
    };

    struct op_pow {
        float y_;
        op_pow( float y ) : y_(y) {}
        inline __device__ float operator()( float a ) const { return powf(a, y_); }
    };

    struct op_log {
        inline __device__ float operator()( float a ) const { return logf(a); }
    };

    template<typename T> struct op_abs_diff : public binary_function<T,T,float> {
        inline __device__ float operator()( T a, T b ) const { return length(a - b); }
    };

    template<typename T> struct op_log_abs : public unary_function<T,float> {
        inline __device__ float operator()( T a ) const { return logf(fmaxf(1e-7, length(a))); }
    };

    template<typename T> struct imp_circshift : public generator<T> {
        gpu_plm2<T> src_;
        int dx_;
        int dy_;

        imp_circshift( const gpu_image& src, int dx, int dy )
            : src_(src), dx_(dx), dy_(dy) {}

        inline __device__ T operator()(int ix, int iy) const {
            return src_((ix - dx_) % src_.w, (iy - dy_) % src_.h);
        }
    };
}


oz::gpu_image oz::operator-( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:  return transform(src, op_neg<float>());
        case FMT_FLOAT2: return transform(src, op_neg<float2>());
        case FMT_FLOAT3: return transform(src, op_neg<float3>());
        case FMT_FLOAT4: return transform(src, op_neg<float4>());
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::operator+( const gpu_image& a, const gpu_image& b ) {
    if (a.format() != b.format()) OZ_INVALID_FORMAT();
    if (a.size() != b.size()) OZ_INVALID_SIZE();
    switch (a.format()) {
        case FMT_FLOAT:  return transform(a, b, op_add<float>());
        case FMT_FLOAT2: return transform(a, b, op_add<float2>());
        case FMT_FLOAT3: return transform(a, b, op_add<float3>());
        case FMT_FLOAT4: return transform(a, b, op_add<float4>());
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::operator+( const gpu_image& a, float b ) {
    return transform_channel_f(a, op_scalar_add(b));
}


oz::gpu_image oz::operator-( const gpu_image& a, const gpu_image& b ) {
    if (a.format() != b.format()) OZ_INVALID_FORMAT();
    if (a.size() != b.size()) OZ_INVALID_SIZE();
    switch (a.format()) {
        case FMT_FLOAT:  return transform(a, b, op_sub<float>());
        case FMT_FLOAT2: return transform(a, b, op_sub<float2>());
        case FMT_FLOAT3: return transform(a, b, op_sub<float3>());
        case FMT_FLOAT4: return transform(a, b, op_sub<float4>());
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::operator-( const gpu_image& a, float b ) {
    return transform_channel_f(a, op_scalar_add(-b));
}


oz::gpu_image oz::operator*( const gpu_image& a, const gpu_image& b ) {
    if ((a.format() != b.format()) && (b.format() != FMT_FLOAT)) OZ_INVALID_FORMAT();
    if (a.size() != b.size()) OZ_INVALID_SIZE();
    if (b.format() == FMT_FLOAT) {
        switch (a.format()) {
            case FMT_FLOAT:  return transform(a, b, op_mul<float ,float>());
            case FMT_FLOAT2: return transform(a, b, op_mul<float2,float>());
            case FMT_FLOAT3: return transform(a, b, op_mul<float3,float>());
            case FMT_FLOAT4: return transform(a, b, op_mul<float4,float>());
            default:
                OZ_INVALID_FORMAT();
        }
    } else {
        switch (a.format()) {
            case FMT_FLOAT2: return transform(a, b, op_mul<float2,float2>());
            case FMT_FLOAT3: return transform(a, b, op_mul<float3,float3>());
            case FMT_FLOAT4: return transform(a, b, op_mul<float4,float4>());
            default:
                OZ_INVALID_FORMAT();
        }
    }
}


oz::gpu_image oz::operator*( const gpu_image& src, float k ) {
    switch (src.format()) {
        case FMT_FLOAT:  return transform(src, op_scalar_mul<float >(k));
        case FMT_FLOAT2: return transform(src, op_scalar_mul<float2>(k));
        case FMT_FLOAT3: return transform(src, op_scalar_mul<float3>(k));
        case FMT_FLOAT4: return transform(src, op_scalar_mul<float4>(k));
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::operator*( float k, const gpu_image& src ) {
    return operator*(src, k);
}


oz::gpu_image oz::operator*( const gpu_image& src, float2 k ) {
    return transform(src, op_vec_mul<float2>(k));
}


oz::gpu_image oz::operator*( float2 k, const gpu_image& src ) {
    return operator*(src, k);
}


oz::gpu_image oz::operator*( const gpu_image& src, float3 k ) {
    return transform(src, op_vec_mul<float3>(k));
}


oz::gpu_image oz::operator*( float3 k, const gpu_image& src ) {
    return operator*(src, k);
}


oz::gpu_image oz::operator*( const gpu_image& src, float4 k ) {
    return transform(src, op_vec_mul<float4>(k));
}


oz::gpu_image oz::operator*( float4 k, const gpu_image& src ) {
    return operator*(src, k);
}


oz::gpu_image oz::operator/( const gpu_image& a, const gpu_image& b ) {
    if ((a.format() != b.format()) && (b.format() != FMT_FLOAT)) OZ_INVALID_FORMAT();
    if (a.size() != b.size()) OZ_INVALID_SIZE();
    if (b.format() == FMT_FLOAT) {
        switch (a.format()) {
            case FMT_FLOAT:  return transform(a, b, op_div<float, float>());
            case FMT_FLOAT2: return transform(a, b, op_div<float2,float>());
            case FMT_FLOAT3: return transform(a, b, op_div<float3,float>());
            case FMT_FLOAT4: return transform(a, b, op_div<float4,float>());
            default:
                OZ_INVALID_FORMAT();
        }
    } else {
        switch (a.format()) {
            case FMT_FLOAT2: return transform(a, b, op_div<float2,float2>());
            case FMT_FLOAT3: return transform(a, b, op_div<float3,float3>());
            case FMT_FLOAT4: return transform(a, b, op_div<float4,float4>());
            default:
                OZ_INVALID_FORMAT();
        }
    }
}


oz::gpu_image oz::operator/( const gpu_image& src, float k ) {
    return operator*(src, 1.0f / k);
}


void oz::gpu_image::fill( float value, int x, int y, int w, int h ) {
    op_fill<float> op(*this, make_int2(x, y), value);
    foreach(w, h, op);
}


void oz::gpu_image::fill( float2 value, int x, int y, int w, int h ) {
    op_fill<float2> op(*this, make_int2(x, y), value);
    foreach(w, h, op);
}


void oz::gpu_image::fill( float3 value, int x, int y, int w, int h ) {
    op_fill<float3> op(*this, make_int2(x, y), value);
    foreach(w, h, op);
}


void oz::gpu_image::fill( float4 value, int x, int y, int w, int h ) {
    op_fill<float4> op(*this, make_int2(x, y), value);
    foreach(w, h, op);
}


oz::gpu_image oz::adjust( const gpu_image& src, float a, float b ) {
    return transform_channel_f(src, op_adjust(a, b));
}


oz::gpu_image oz::invert( const gpu_image& src ) {
    return transform_channel_f(src, op_invert());
}


oz::gpu_image oz::saturate( const gpu_image& src ) {
    return transform_channel_f(src, op_saturate());
}


oz::gpu_image oz::clamp( const gpu_image& src, float a, float b ) {
    return transform_channel_f(src, op_clamp(a, b));
}


oz::gpu_image oz::clamp( const gpu_image& src, float2 a, float2 b ) {
    return transform(src, op_vec_clamp<float2>(a, b));
}


oz::gpu_image oz::clamp( const gpu_image& src, float3 a, float3 b ) {
    return transform(src, op_vec_clamp<float3>(a, b));
}


oz::gpu_image oz::clamp( const gpu_image& src, float4 a, float4 b ) {
    return transform(src, op_vec_clamp<float4>(a, b));
}


oz::gpu_image oz::lerp( const gpu_image& src0, const gpu_image& src1, float t ) {
    return transform_channel_f(src0, src1, op_lerp(t));
}


oz::gpu_image oz::abs( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:  return transform(src, op_abs<float >());
        case FMT_FLOAT2: return transform(src, op_abs<float2>());
        case FMT_FLOAT3: return transform(src, op_abs<float3>());
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::abs2( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:  return transform(src, op_abs2<float >());
        case FMT_FLOAT2: return transform(src, op_abs2<float2>());
        case FMT_FLOAT3: return transform(src, op_abs2<float3>());
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::sqrt( const gpu_image& src ) {
    return transform_channel_f(src, op_sqrt());
}


oz::gpu_image oz::sqr( const gpu_image& src ) {
    return transform_channel_f(src, op_sqr());
}


oz::gpu_image oz::pow( const gpu_image& src, float y ) {
    return transform_channel_f(src, op_pow(y));
}


oz::gpu_image oz::log( const gpu_image& src ) {
    return transform_channel_f(src, op_log());
}


oz::gpu_image oz::abs_diff( const gpu_image& src0, const gpu_image& src1 ) {
    switch (src0.format()) {
        case FMT_FLOAT:  return transform(src0, src1, op_abs_diff<float >());
        case FMT_FLOAT2: return transform(src0, src1, op_abs_diff<float2>());
        case FMT_FLOAT3: return transform(src0, src1, op_abs_diff<float3>());
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::log_abs( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:  return transform(src, op_log_abs<float >());
        case FMT_FLOAT2: return transform(src, op_log_abs<float2>());
        case FMT_FLOAT3: return transform(src, op_log_abs<float3>());
        default:
            OZ_INVALID_FORMAT();
    }
}



oz::gpu_image oz::circshift( const gpu_image& src, int dx, int dy ) {
    switch (src.format()) {
        case FMT_FLOAT:  return generate(src.size(), imp_circshift<float >(src, dx, dy));
        case FMT_FLOAT2: return generate(src.size(), imp_circshift<float2>(src, dx, dy));
        case FMT_FLOAT3: return generate(src.size(), imp_circshift<float3>(src, dx, dy));
        case FMT_FLOAT4: return generate(src.size(), imp_circshift<float4>(src, dx, dy));
        default:
            OZ_INVALID_FORMAT();
    }
}

