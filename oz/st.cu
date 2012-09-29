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
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/generate.h>
#include <oz/transform.h>
#include <oz/gpu_sampler1.h>
#include <oz/gauss.h>


namespace oz {

    template<typename T> struct StCentralDiff : public generator<float3> {
        gpu_sampler<T,0> src_;

        StCentralDiff( const gpu_image& src ) : src_(src) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            T u = ( src_(ix+1, iy  ) - src_(ix-1, iy ) ) / 2;
            T v = ( src_(ix,   iy+1) - src_(ix,   iy-1) ) / 2;
            return make_float3(dot(u, u), dot(v, v), dot(u, v));
        }
    };

    gpu_image st_central_diff( const gpu_image& src ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), StCentralDiff<float >(src));
            case FMT_FLOAT3: return generate(src.size(), StCentralDiff<float3>(src));
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T> struct StGaussian : public generator<float3> {
        gpu_sampler<T,0> src_;
        float rho_;
        float precision_;
        bool normalize_;

        StGaussian( const gpu_image& src, float rho, float precision, bool normalize )
            : src_(src), rho_(rho), precision_(precision), normalize_(normalize) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float twoRho2 = 2.0f * rho_ * rho_;
            int halfWidth = int(ceilf( precision_ * rho_ + 1 ));

            T u = make_zero<T>();
            T v = make_zero<T>();

            if (halfWidth > 0) {
                float sum = 0;
                for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                    for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                        float d = length(make_float2(i,j));
                        float e = __expf( -d *d / twoRho2 );

                        T c = src_(ix + i, iy + j);
                        u += i * e * c;
                        v += j * e * c;

                        sum += e;
                    }
                }

                sum *= -rho_ * rho_;
                u /= sum;
                v /= sum;
            }

            if (normalize_) {
                return st_normalized(u, v);
            } else {
                return make_float3(dot(u,u), dot(v,v), dot(u,v));
            }
        }
    };

    gpu_image st_gaussian( const gpu_image& src, float rho, float precision, bool normalize ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), StGaussian<float >(src, rho, precision, normalize));
            case FMT_FLOAT3: return generate(src.size(), StGaussian<float3>(src, rho, precision, normalize));
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T> struct StGaussianX2 : public generator<float3> {
        gpu_sampler<T,0> src_;
        float rho_;
        float precision_;
        bool normalize_;

        StGaussianX2( const gpu_image& src, float rho, float precision, bool normalize )
            : src_(src), rho_(rho), precision_(precision), normalize_(normalize) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float twoRho2 = 2.0f * rho_ * rho_ * 4;
            int halfWidth = 2 * int(ceilf( precision_ * rho_ + 1 ));

            T u = make_zero<T>();
            T v = make_zero<T>();

            if (halfWidth > 0) {
                float sum = 0;
                for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                    for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                        float d = length(make_float2(i,j));
                        float e = __expf( -d *d / twoRho2 );

                        T c = src_(0.5f * (ix + i), 0.5f * (iy + j));
                        u += i * e * c;
                        v += j * e * c;

                        sum += e;
                    }
                }

                sum *= -rho_ * rho_;
                u /= sum;
                v /= sum;
            }

            if (normalize_) {
                return st_normalized(u, v);
            } else {
                return make_float3(dot(u,u), dot(v,v), dot(u,v));
            }
        }
    };

    gpu_image st_gaussian_x2( const gpu_image& src, float rho, float precision, bool normalize ) {
        int w = 2 * src.w();
        int h = 2 * src.h();
        switch (src.format()) {
            case FMT_FLOAT:  return generate(w, h, StGaussianX2<float >(src, rho, precision, normalize));
            case FMT_FLOAT3: return generate(w, h, StGaussianX2<float3>(src, rho, precision, normalize));
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T> struct StSobel : public generator<float3> {
        gpu_sampler<T,0> src_;

        StSobel( const gpu_image& src ) : src_(src) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            T u = (
                -1 * src_(ix - 1, iy - 1) +
                -2 * src_(ix - 1, iy    ) +
                -1 * src_(ix - 1, iy + 1) +
                +1 * src_(ix + 1, iy - 1) +
                +2 * src_(ix + 1, iy    ) +
                +1 * src_(ix + 1, iy + 1)
                ) / 8;

            T v = (
                -1 * src_(ix - 1, iy - 1) +
                -2 * src_(ix,     iy - 1) +
                -1 * src_(ix + 1, iy - 1) +
                +1 * src_(ix - 1, iy + 1) +
                +2 * src_(ix,     iy + 1) +
                +1 * src_(ix + 1, iy + 1)
                ) / 8;

            return make_float3(dot(u, u), dot(v, v), dot(u, v));
        }
    };

    gpu_image st_sobel( const gpu_image& src, float rho ) {
        gpu_image dst;
        switch (src.format()) {
            case FMT_FLOAT:
                dst = generate(src.size(), StSobel<float >(src));
                break;
            case FMT_FLOAT3:
                dst = generate(src.size(), StSobel<float3>(src));
                break;
            default:
                OZ_INVALID_FORMAT();
        }
        return (rho == 0)? dst : gauss_filter_xy(dst, rho);
    }


    template<typename T> struct StScharr3x3 : public generator<float3> {
        gpu_sampler<T,0> src_;
        bool normalize_;

        StScharr3x3( const gpu_image& src, bool normalize )
            : src_(src), normalize_(normalize) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            const float p1 = 46.84f / 256;
            const float p2 = 1 - 2 * p1;
            T u = (
                -p1 * src_(ix - 1, iy - 1) +
                -p2 * src_(ix - 1, iy    ) +
                -p1 * src_(ix - 1, iy + 1) +
                +p1 * src_(ix + 1, iy - 1) +
                +p2 * src_(ix + 1, iy    ) +
                +p1 * src_(ix + 1, iy + 1)
                ) / 2;

            T v = (
                -p1 * src_(ix - 1, iy - 1) +
                -p2 * src_(ix,     iy - 1) +
                -p1 * src_(ix + 1, iy - 1) +
                +p1 * src_(ix - 1, iy + 1) +
                +p2 * src_(ix,     iy + 1) +
                +p1 * src_(ix + 1, iy + 1)
                ) / 2;

            if (normalize_) {
                return st_normalized(u, v);
            } else {
                return make_float3(dot(u,u), dot(v,v), dot(u,v));
            }
        }
    };

    gpu_image st_scharr_3x3( const gpu_image& src, float rho, bool normalize ) {
        gpu_image dst;
        switch (src.format()) {
            case FMT_FLOAT:
                dst = generate(src.size(), StScharr3x3<float>(src, normalize));
                break;
            case FMT_FLOAT3:
                dst = generate(src.size(), StScharr3x3<float3>(src, normalize));
                break;
            default:
                OZ_INVALID_FORMAT();
        }
        return (rho == 0)? dst : gauss_filter_xy(dst, rho);
    }


    template<typename T> struct StScharr5x5 : public generator<float3> {
        gpu_sampler<T,0> src_;
        bool normalize_;

        StScharr5x5( const gpu_image& src, bool normalize )
            : src_(src), normalize_(normalize) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            const float d1 = 21.27f / 256;
            const float d2 = 85.46f / 256;
            const float p1 = 5.91f / 256;
            const float p2 = 61.77f / 256;
            const float p3 = 1 - 2 * p1 - 2 * p2;

            float K[5][5] = {
                {d1*p1, d2*p1, 0, -d2*p1, -d1*p1},
                {d1*p2, d2*p2, 0, -d2*p2, -d1*p2},
                {d1*p3, d2*p3, 0, -d2*p3, -d1*p3},
                {d1*p2, d2*p2, 0, -d2*p2, -d1*p2},
                {d1*p1, d2*p1, 0, -d2*p1, -d1*p1}
            };

            T u = make_zero<T>();
            T v = make_zero<T>();
            for (int j = -2; j <= 2; ++j) {
                for (int i = -2; i <= 2; ++i) {
                    T c = src_(ix + i, iy + j);
                    u += K[2-j][2-i] * c;
                    v += K[2-i][2-j] * c;
                }
            }

            if (normalize_) {
                return st_normalized(u, v);
            } else {
                return make_float3(dot(u,u), dot(v,v), dot(u,v));
            }
        }
    };

    gpu_image st_scharr_5x5( const gpu_image& src, float rho, bool normalize ) {
        gpu_image dst;
        switch (src.format()) {
            case FMT_FLOAT:
                dst = generate(src.size(), StScharr5x5<float>(src, normalize));
                break;
            default:
                OZ_INVALID_FORMAT();
        }
        return (rho == 0)? dst : gauss_filter_xy(dst, rho);
    }


    struct StFromGradient : public unary_function<float2,float3> {
        inline __device__ float3 operator()( float2 g ) const {
            return make_float3(g.x * g.x, g.y * g.y, g.x * g.y);
        }
    };

    gpu_image st_from_gradient( const gpu_image& g ) {
        return transform(g, StFromGradient());
    }


    struct StFromTangent : public unary_function<float2,float3> {
        inline __device__ float3 operator()( float2 t ) const {
            float2 g = make_float2(t.y, -t.x);
            return make_float3(g.x * g.x, g.y * g.y, g.x * g.y);
        }
    };

    gpu_image st_from_tangent( const gpu_image& t ) {
        return transform(t, StFromTangent());
    }


    struct StToTangent : public unary_function<float3,float2> {
        inline __device__ float2 operator()( float3 g ) const {
            return st2tangent(g);
        }
    };

    gpu_image st_to_tangent( const gpu_image& st ) {
        return transform(st, StToTangent());
    }


    struct StToGradient : public unary_function<float3,float2> {
        inline __device__ float2 operator()( float3 g ) const {
            return st2gradient(g);
        }
    };

    gpu_image st_to_gradient( const gpu_image& st ) {
        return transform(st, StToGradient());
    }


    struct StLFM : public unary_function<float3,float4> {
        float alpha_;

        StLFM( float alpha ) : alpha_(alpha) {}

        inline __device__ float4 operator()( float3 g ) const {
            return (alpha_ == 0)? st2lfm(g) : st2lfm(g, alpha_);
        }
    };

    gpu_image st_lfm( const gpu_image& st, float alpha ) {
        return transform(st, StLFM(alpha));
    }


    struct StLFM2 : public unary_function<float3,float4> {
        moa_t moa_;
        float alpha_;

        StLFM2( moa_t moa, float alpha ) : moa_(moa), alpha_(alpha) {}

        inline __device__ float4 operator()( float3 g ) const {
            return (alpha_ == 0)? st2lfm(g) : st2lfm(g, moa_, alpha_);
        }
    };

    gpu_image st_lfm2( const gpu_image& st, moa_t moa, float alpha ) {
        return transform(st, StLFM2(moa, alpha));
    }


    struct StAngle : public unary_function<float3,float> {
        inline __device__ float operator()( float3 g ) const {
            return st2angle(g);
        }
    };

    gpu_image st_angle( const gpu_image& st ) {
        return transform(st, StAngle());
    }


    struct StMoa : public unary_function<float3,float> {
        moa_t moa_;
        StMoa( moa_t moa ) : moa_(moa) {}
        inline __device__ float operator()( float3 g ) const {
            return st2moa(g, moa_);
        }
    };

    gpu_image st_moa( const gpu_image& st, moa_t moa ) {
        return transform(st, StMoa(moa));
    }


    struct StNormalize : public unary_function<float3,float3> {
        inline __device__ float3 operator()( float3 g ) const {
            float mag = sqrtf(fmaxf(0, g.x * g.x + g.y * g.y + 2 * g.z * g.z));
            if (mag > 0)
                return make_float3(g.x / mag, g.y / mag, g.z / mag);
            else
                return make_float3(0);
        }
    };

    gpu_image st_normalize( const gpu_image& st ) {
        return transform(st, StNormalize());
    }


    struct StFlatten : public unary_function<float3,float3> {
        inline __device__ float3 operator()( float3 g ) const {
            float phi = 0.5f * atan2(2 * g.z, g.x - g.y);
            float a = 0.5f * (g.y + g.x);
            float b = 0.5f * sqrtf(fmaxf(0.0, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));
            float2 l = make_float2(a + b, a - b);

            float c = cosf(phi);
            float s = sinf(phi);

            return make_float3(
                l.x*c*c,
                l.x*s*s,
                l.x*c*s
            );
        }
    };

    gpu_image st_flatten( const gpu_image& st ) {
        return transform(st, StFlatten());
    }


    struct StRotate : public unary_function<float3,float3> {
        float s_, c_;

        StRotate( float angle ) : s_(sinf(angle)), c_(cosf(angle)) {}

        inline __device__ float3 operator()( float3 g ) const {
            return make_float3(
                c_*c_*g.x + 2*s_*c_*g.z + s_*s_*g.y,
                s_*s_*g.x - 2*s_*c_*g.z + c_*c_*g.y,
                (c_*c_ - s_*s_)*g.z + c_*s_*(g.y - g.x)
            );
        }
    };

    gpu_image st_rotate( const gpu_image& st, float angle ) {
        return transform(st, StRotate(radians(angle)));
    }


    struct StPow : public unary_function<float3,float3> {
        float y_;

        StPow( float y ) : y_(y) {}

        inline __device__ float3 operator()( float3 g ) const {
            return make_float3( powf(fmaxf(0, g.x), y_), powf(fmaxf(0, g.y), y_), sign(g.z)*powf(fabs(g.z), y_) );
        }
    };

    gpu_image st_pow( const gpu_image& st, float y ) {
        return transform(st, StPow(y));
    }


    struct StExp : public unary_function<float3,float3> {
        inline __device__ float3 operator()( float3 g ) const {
            /*
            float4 g = st(ix, iy);

            float phi = 0.5f * atan2(2 * g.z, g.x - g.y);
            float a = 0.5f * (g.y + g.x);
            float b = 0.5f * sqrtf(max(0.0, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));

            float l1 = a + b;
            float l2 = a - b;
            float2 l = make_float2(exp(l1), exp(l2));

            float c = cosf(phi);
            float s = sinf(phi);

            dst(ix, iy) = make_float4(
                l.x*c*c + l.y*s*s,
                l.x*s*s + l.y*c*c,
                l.x*c*s - l.y*c*s,
                1
            );
            */

            if (fabs(g.z) > 0) {
                float tr2 = (g.x + g.y) / 2;
                float det = g.x * g.y - g.z * g.z;
                float s = sqrtf( fmaxf(0, tr2 * tr2 - det) );

                float2 l = make_float2(expf(tr2 - s), expf(tr2 + s));
                float2 e1, e2;

                float eg2 = (g.x - g.y) / 2;
                if (eg2 < 0) {
                    e1 = normalize(make_float2(+eg2 - s, g.z));
                    e2 = make_float2(e1.y, -e1.x);
                } else {
                    e1 = normalize(make_float2(g.z, -eg2 - s));
                    e2 = make_float2(-e1.y, e1.x);
                }

                return make_float3(
                    l.x * e1.x*e1.x + l.y * e2.x*e2.x,
                    l.x * e1.y*e1.y + l.y * e2.y*e2.y,
                    l.x * e1.x*e1.y + l.y * e2.x*e2.y
                );
            } else {
                return make_float3(expf(g.x), expf(g.y), 0);
            }
        }
    };

    gpu_image st_exp( const gpu_image& st ) {
        return transform(st, StExp());
    }


    struct StLog : public unary_function<float3,float3> {
        inline __device__ float3 operator()( float3 g ) const {
            /*
            float4 g = st(ix, iy);

            float phi = 0.5f * atan2(2 * g.z, g.x - g.y);
            float a = 0.5f * (g.y + g.x);
            float b = 0.5f * sqrtf(max(0.0, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));

            float l1 = max(0.00001, a + b);
            float l2 = max(0.00001, a - b);
            float2 l = make_float2(log(l1), log(l2));

            float c = cosf(phi);
            float s = sinf(phi);

            dst(ix, iy) = make_float4(
                l.x*c*c + l.y*s*s,
                l.x*s*s + l.y*c*c,
                l.x*c*s - l.y*c*s,
                1
            );
            */

            if (fabs(g.z) > 0) {
                float tr2 = (g.x + g.y) / 2;
                float det = g.x * g.y - g.z * g.z;
                float s = sqrtf( fmaxf(0, tr2 * tr2 - det) );

                float l1 = fmaxf(1e-7, tr2 - s);
                float l2 = fmaxf(1e-7, tr2 + s);
                float2 l = make_float2(logf(l1), logf(l2));
                float2 e1, e2;

                float eg2 = (g.x - g.y) / 2;
                if (eg2 < 0) {
                    e1 = normalize(make_float2(+eg2 - s, g.z));
                    e2 = make_float2(e1.y, -e1.x);
                } else {
                    e1 = normalize(make_float2(g.z, -eg2 - s));
                    e2 = make_float2(-e1.y, e1.x);
                }

                return make_float3(
                    l.x * e1.x*e1.x + l.y * e2.x*e2.x,
                    l.x * e1.y*e1.y + l.y * e2.y*e2.y,
                    l.x * e1.x*e1.y + l.y * e2.x*e2.y
                );
            } else {
                float l1 = fmaxf(1e-7, g.x);
                float l2 = fmaxf(1e-7, g.y);
                float2 l = make_float2(logf(l1), logf(l2));
                return make_float3(l.x, l.y, 0);
            }
        }
    };

    gpu_image st_log( const gpu_image& st ) {
        return transform(st, StLog());
    }

}
