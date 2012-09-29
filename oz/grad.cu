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
#include <oz/grad.h>
#include <oz/generate.h>
#include <oz/transform.h>
#include <oz/gpu_sampler1.h>


namespace oz {

    struct GradGaussian : public generator<float2> {
        gpu_sampler<float,0> src_;
        float sigma_;
        float precision_;
        bool normalize_;

        GradGaussian( const gpu_image& src, float sigma, float precision, bool normalize )
            : src_(src), sigma_(sigma), precision_(precision), normalize_(normalize) {}

        inline __device__ float2 operator()( int ix, int iy ) const {
            float twoSigma2 = 2.0f * sigma_ * sigma_;
            int halfWidth = int(ceilf( precision_ * sigma_ + 1 ));

            float gx = 0;
            float gy = 0;
            float2 result = make_float2(0);

            if (halfWidth > 0) {
                float sum = 0;
                for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                    for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                        float d = length(make_float2(i,j));
                        float e = __expf( -d *d / twoSigma2 );

                        float c = src_(ix + i, iy + j);
                        gx += i * e * c;
                        gy += j * e * c;

                        sum += e;
                    }
                }

                if (normalize_) {
                    float n = sqrtf(gx*gx + gy*gy);
                    if (n > 0) result = make_float2(gx / n, gy / n);
                } else {
                    sum *= sigma_ * sigma_;
                    gx /= sum;
                    gy /= sum;
                    result= make_float2(gx, gy);
                }
            }

            return result;
        }
    };

    gpu_image grad_gaussian( const gpu_image& src, float sigma, float precision, bool normalize ) {
        return generate(src.size(), GradGaussian(src, sigma, precision, normalize));
    }


    struct GradCentralDiff : public oz::generator<float2> {
        gpu_sampler<float,0> src_;
        bool normalize_;

        GradCentralDiff( const oz::gpu_image& src, bool normalize )
            : src_(src), normalize_(normalize) {}

        __device__ float2 operator()( int ix, int iy) const {
            float gx = ( src_(ix+1, iy  ) - src_(ix-1, iy  ) ) / 2;
            float gy = ( src_(ix,   iy+1) - src_(ix,   iy-1) ) / 2;

            float2 result;
            if (normalize_) {
                float n = sqrtf(gx*gx + gy*gy);
                result = (n > 0)? make_float2(gx / n, gy / n) : make_float2(0);
            } else {
                result= make_float2(gx, gy);
            }
            return result;
        }
    };

    gpu_image grad_central_diff( const gpu_image& src, bool normalize ) {
        return generate(src.size(), GradCentralDiff(src, normalize));
    }


    struct GradSobel : public oz::generator<float2> {
        gpu_sampler<float,0> src_;
        bool normalize_;

        GradSobel( const oz::gpu_image& src, bool normalize )
            : src_(src), normalize_(normalize) {}

        __device__ float2 operator()( int ix, int iy) const {
            float gx = (
                -1 * src_(ix-1, iy-1) +
                -2 * src_(ix-1, iy) +
                -1 * src_(ix-1, iy+1) +
                +1 * src_(ix+1, iy-1) +
                +2 * src_(ix+1, iy) +
                +1 * src_(ix+1, iy+1)
                ) / 8;

            float gy = (
                -1 * src_(ix-1, iy-1) +
                -2 * src_(ix,   iy-1) +
                -1 * src_(ix+1, iy-1) +
                +1 * src_(ix-1, iy+1) +
                +2* src_(ix,   iy+1) +
                +1 * src_(ix+1, iy+1)
                ) / 8;

            float2 result;
            if (normalize_) {
                float n = sqrtf(gx*gx + gy*gy);
                result = (n > 0)? make_float2(gx / n, gy / n) : make_float2(0);
            } else {
                result = make_float2(gx, gy);
            }
            return result;
        }
    };

    gpu_image grad_sobel( const gpu_image& src, bool normalize ) {
        return generate(src.size(), GradSobel(src, normalize));
    }


    struct GradScharr3x3 : public oz::generator<float2> {
        gpu_sampler<float,0> src_;
        bool normalize_;

        GradScharr3x3( const oz::gpu_image& src, bool normalize )
            : src_(src), normalize_(normalize) {}

        __device__ float2 operator()( int ix, int iy) const {
            const float p1 = 46.84f / 256;
            const float p2 = 1 - 2 * p1;
            float gx = (
                -p1 * src_(ix-1, iy-1) +
                -p2 * src_(ix-1, iy) +
                -p1 * src_(ix-1, iy+1) +
                 p1 * src_(ix+1, iy-1) +
                 p2 * src_(ix+1, iy) +
                 p1 * src_(ix+1, iy+1)
                ) / 2;

            float gy = (
                -p1 * src_(ix-1, iy-1) +
                -p2 * src_(ix,   iy-1) +
                -p1 * src_(ix+1, iy-1) +
                +p1 * src_(ix-1, iy+1) +
                +p2 * src_(ix,   iy+1) +
                +p1 * src_(ix+1, iy+1)
                ) / 2;

            if (normalize_) {
                float n = sqrtf(gx*gx + gy*gy);
                return (n > 0)? make_float2(gx / n, gy / n) : make_float2(0);
            } else {
                return make_float2(gx, gy);
            }
        }
    };

    gpu_image grad_scharr_3x3( const gpu_image& src, bool normalize ) {
        return generate(src.size(), GradScharr3x3(src, normalize));
    }


    struct GradScharr5x5 : public generator<float2> {
        gpu_sampler<float,0> src_;
        bool normalize_;

        GradScharr5x5( const gpu_image& src, bool normalize )
            : src_(src), normalize_(normalize) {}

        inline __device__ float2 operator()( int ix, int iy ) const {
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

            float gx = 0;
            float gy = 0;
            for (int j = -2; j <= 2; ++j) {
                for (int i = -2; i <= 2; ++i) {
                    float c = src_(ix + i, iy + j);
                    gx += K[2-j][2-i] * c;
                    gy += K[2-i][2-j] * c;
                }
            }

            if (normalize_) {
                float n = sqrtf(gx*gx + gy*gy);
                return  (n > 0)? make_float2(gx / n, gy / n) : make_float2(0);
            } else {
                return make_float2(gx, gy);
            }
        }
    };

    gpu_image grad_scharr_5x5( const gpu_image& src, bool normalize ) {
        return generate(src.size(), GradScharr5x5(src, normalize));
    }


    struct GradToAxis : public oz::unary_function<float2,float2> {
        bool squared_;

        GradToAxis( bool squared )
            : squared_(squared) {}

        __device__ float2 operator()( float2 g ) const {
            float n = g.x*g.x + g.y*g.y;
            if (n > 0) {
                if (!squared_) n = sqrtf(n);
                float phi = 2 * atan2(g.y, g.x);
                return make_float2(n * __cosf(phi), n * __sinf(phi));
            } else {
                return make_float2(0);
            }
        }
    };

    gpu_image grad_to_axis( const gpu_image& src, bool squared ) {
        return transform(src, GradToAxis(squared));
    }


    struct GradFromAxis : public oz::unary_function<float2,float2> {
        bool squared_;

        GradFromAxis( bool squared )
            : squared_(squared) {}

        __device__ float2 operator()( float2 g ) const {
            float n = g.x*g.x + g.y*g.y;
            if (n > 0) {
                if (!squared_) n = sqrtf(n);
                float phi = 0.5f * atan2(g.y / n, g.x / n);
                return make_float2(n * __cosf(phi), n * __sinf(phi));
            } else {
                return make_float2(0);
            }
        }
    };

    gpu_image grad_from_axis( const gpu_image& src, bool squared ) {
        return transform(src, GradFromAxis(squared));
    }


    struct GradAngle : public oz::unary_function<float2,float> {
        bool perpendicular_;

        GradAngle( bool perpendicular ) : perpendicular_(perpendicular) {}

        __device__ float operator()( float2 g ) const {
            return perpendicular_? atan2(-g.x, g.y) : atan2(g.y, g.x);
        }
    };

    gpu_image grad_angle( const gpu_image& src, bool perpendicular ) {
        return transform(src, GradAngle(perpendicular));
    }


    struct GradToLFM : public oz::unary_function<float2,float4> {
        __device__ float4 operator()( float2 g ) const {
            return make_float4( -g.y, g.x, 1, 1 );
        }
    };

    gpu_image grad_to_lfm( const gpu_image& src ) {
        return transform(src, GradToLFM());
    }


    template<typename T> struct GradSobelMag : public oz::generator<float> {
        gpu_sampler<T,0> src_;

        GradSobelMag( const oz::gpu_image& src ) : src_(src) {}

        __device__ float operator()( int ix, int iy ) const {
            T u = (
                -1 * src_(ix-1, iy-1) +
                -2 * src_(ix-1, iy  ) +
                -1 * src_(ix-1, iy+1) +
                +1 * src_(ix+1, iy-1) +
                +2 * src_(ix+1, iy  ) +
                +1 * src_(ix+1, iy+1)
                );

            T v = (
                -1 * src_(ix-1, iy-1) +
                -2 * src_(ix,   iy-1) +
                -1 * src_(ix+1, iy-1) +
                +1 * src_(ix-1, iy+1) +
                +2 * src_(ix,   iy+1) +
                +1 * src_(ix+1, iy+1)
                );

            return sqrtf(dot(u,u) + dot(v,v));
        }
    };

    gpu_image grad_sobel_mag( const gpu_image& src ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), GradSobelMag<float >(src));
            case FMT_FLOAT3: return generate(src.size(), GradSobelMag<float3>(src));
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
