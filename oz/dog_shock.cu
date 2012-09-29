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
#include <oz/dog_shock.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>
#include <oz/deriv2nd.h>
#include <cfloat>

namespace oz {

    inline __device__ float max3(float a, float b, float c) {
        return max(a, max(b,c));
    }

    inline __device__ float min3(float a, float b, float c) {
        return min(a, min(b,c));
    }


    struct DogShockUpwind : public generator<float3> {
        const gpu_sampler<float3,0> src_;
        const gpu_plm2<float> dog_;
        float step_;

        DogShockUpwind( const gpu_image& src, const gpu_image& dog, float step )
            : src_(src), dog_(dog), step_(step) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float sign = dog_(ix, iy);
            float3 cij = src_(ix, iy);

            if (sign > 0) {
                float3 cxl = src_(ix - 1, iy);
                float3 cxr = src_(ix + 1, iy);
                float rx = max3(cxr.x - cij.x, cxl.x - cij.x, 0);
                float gx = max3(cxr.y - cij.y, cxl.y - cij.y, 0);
                float bx = max3(cxr.z - cij.z, cxl.z - cij.z, 0);

                float3 cyb = src_(ix, iy + 1);
                float3 cyt = src_(ix, iy - 1);
                float ry = max3(cyt.x - cij.x, cyb.x - cij.x, 0);
                float gy = max3(cyt.y - cij.y, cyb.y - cij.y, 0);
                float by = max3(cyt.z - cij.z, cyb.z - cij.z, 0);

                cij.x += step_ * sqrtf( rx*rx + ry*ry );
                cij.y += step_ * sqrtf( gx*gx + gy*gy );
                cij.z += step_ * sqrtf( bx*bx + by*by );
            }
            else if (sign < 0) {
                float3 cxl = src_(ix - 1, iy);
                float3 cxr = src_(ix + 1, iy);
                float rx = min3(cxr.x - cij.x, cxl.x - cij.x, 0);
                float gx = min3(cxr.y - cij.y, cxl.y - cij.y, 0);
                float bx = min3(cxr.z - cij.z, cxl.z - cij.z, 0);

                float3 cyb = src_(ix, iy + 1);
                float3 cyt = src_(ix, iy - 1);
                float ry = min3(cyt.x - cij.x, cyb.x - cij.x, 0);
                float gy = min3(cyt.y - cij.y, cyb.y - cij.y, 0);
                float by = min3(cyt.z - cij.z, cyb.z - cij.z, 0);

                cij.x -= step_ * sqrtf( rx*rx + ry*ry );
                cij.y -= step_ * sqrtf( gx*gx + gy*gy );
                cij.z -= step_ * sqrtf( bx*bx + by*by );
            }

            return cij;
        }
    };

    gpu_image dog_shock_upwind( const gpu_image& src, const gpu_image& dog, float step ) {
        return generate(src.size(), DogShockUpwind(src, dog, step));
    }


    struct DogShockFastMinmax : public generator<float3> {
        const gpu_sampler<float3,0> src_;
        const gpu_plm2<float> dog_;
        float radius_;

        DogShockFastMinmax( const gpu_image& src, const gpu_image& dog, float radius )
            : src_(src), dog_(dog), radius_(radius) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float sign = dog_(ix, iy);
            int r = ceil(radius_);
            float r2 = radius_ * radius_;

            if (sign > 0) {
                float max_sum = FLT_MIN;
                float3 max_val;

                for (int j = -r; j <= r; ++j) {
                    for (int i = -r; i <= r; ++i) {
                        if (i*i+j*j <= r2) {
                            float3 c = src_(ix + i, iy + j);
                            float sum = c.x + c.y + c.z;
                            if (sum > max_sum) {
                                max_sum = sum;
                                max_val = c;
                            }
                        }
                    }
                }
                return max_val;
            } else if (sign < 0) {
                float min_sum = FLT_MAX;
                float3 min_val;

                for (int j = -r; j <= r; ++j) {
                    for (int i = -r; i <= r; ++i) {
                        if (i*i+j*j <= r2) {
                            float3 c = src_(ix + i, iy + j);
                            float sum = c.x + c.y + c.z;
                            if (sum < min_sum) {
                                min_sum = sum;
                                min_val = c;
                            }
                        }
                    }
                }
                return min_val;
            } else {
                return src_(ix, iy);
            }
        }
    };

    gpu_image dog_shock_fast_minmax( const gpu_image& src, const gpu_image& dog, float radius ) {
        return generate(src.size(), DogShockFastMinmax(src, dog, radius));
    }


    struct minmax_gray_t {
        float max_sum;
        float min_sum;
        float3 max_val;
        float3 min_val;
        float gray_sum;
        float gray_N;

        __device__ void init(float3 c) {
            float sum = 0.299f * abs(c.z) + 0.587f * abs(c.y) + 0.114f * abs(c.x);
            max_val = min_val = c;
            max_sum = min_sum = gray_sum = sum;
            gray_N = 1;
        }

        __device__ void add(float3 c) {
            float sum = 0.299f * abs(c.z) + 0.587f * abs(c.y) + 0.114f * abs(c.x);
            gray_sum += sum;
            gray_N += 1;
            if (sum > max_sum) {
                max_sum = sum;
                max_val = c;
            }
            if (sum < min_sum) {
                min_sum = sum;
                min_val = c;
            }
        }

        __device__ float gray_mean() { return gray_sum / gray_N; }
    };

    struct DogShockGradientMinmax : public generator<float3> {
        const gpu_sampler<float3,0> src_;
        const gpu_plm2<float2> tm_;
        const gpu_plm2<float> dog_;
        float radius_;

        DogShockGradientMinmax( const gpu_image& src, const gpu_image& tm, const gpu_image& dog, float radius )
            : src_(src, cudaFilterModeLinear), tm_(tm), dog_(dog), radius_(radius) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float2 t = tm_(ix, iy);
            float2 n = make_float2(t.y, -t.x);
            minmax_gray_t mm;

            float sign = dog_(ix, iy);
            float3 c0 = src_(ix + 0.5f, iy + 0.5f);
            mm.init(c0);

            if (dot(n,n) > 0) {
                float2 nabs = fabs(n);
                float ds;
                float2 dp;
                if (nabs.x > nabs.y) {
                    ds = 1.0f / nabs.x;
                    dp = make_float2(0,0.5f);
                } else {
                    ds = 1.0f / nabs.y;
                    dp = make_float2(0.5f,0);
                }

                float2 uv = make_float2(ix + 0.5f, iy + 0.5f);
                for( float d = ds; d <= radius_; d += ds ) {
                    float2 o = d*n;
                    float2 q;

                    q = make_float2(uv.x + o.x + dp.x, uv.y + o.y + dp.y);
                    {
                        float3 c = src_(q.x, q.y);
                        mm.add(c);
                    }

                    q = make_float2(uv.x + o.x - dp.x, uv.y + o.y - dp.y);
                    {
                        float3 c = src_(q.x, q.y);
                        mm.add(c);
                    }

                    q = make_float2(uv.x - o.x + dp.x, uv.y - o.y + dp.x);
                    {
                        float3 c = src_(q.x, q.y);
                        mm.add(c);
                    }

                    q = make_float2(uv.x - o.x - dp.x, uv.y - o.y - dp.x);
                    {
                        float3 c = src_(q.x, q.y);
                        mm.add(c);
                    }
                }
            }

            return (sign > 0)? mm.max_val : ((sign < 0)? mm.min_val : c0);
        }
    };

    gpu_image dog_shock_gradient_minmax( const gpu_image& src, const gpu_image& tm,
                                     const gpu_image& dog, float radius)
    {
        return generate(src.size(), DogShockGradientMinmax(src, tm, dog, radius));
    }
}