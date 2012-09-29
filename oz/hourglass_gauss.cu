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
#include <oz/hourglass_gauss.h>
#include <oz/gpu_plm2.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {
    template<typename T> struct imp_hourglass_gauss : public oz::generator<T> {
        gpu_sampler<T,0> src_;
        oz::gpu_plm2<float2> tm_;
        float sigma_;
        float rho_;
        float precision_;

        imp_hourglass_gauss( const oz::gpu_image& src, const oz::gpu_image& tm, float sigma, float rho, float precision )
            : src_(src), tm_(tm), sigma_(sigma), rho_(rho), precision_(precision) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            int halfWidth = int(ceilf( precision_ * sigma_ ));
            float twoSigma2 = 2 * sigma_ * sigma_;
            float twoRho2 = 2 * rho_ * rho_;

            float2 t = tm_(ix, iy);
            float2 n = make_float2(t.x, t.y);
            float2 m = make_float2(t.y, -t.x);

            T sum = make_zero<T>();
            float norm = 0;
            if (dot(n,n) > 0) {
                for (int j = -halfWidth; j <= halfWidth; ++j) {
                    for (int i = -halfWidth; i <= halfWidth; ++i) {

                        float2 x = make_float2(i,j);
                        float nx = dot(n,x);
                        float mx = dot(m,x);
                        float kernel;
                        if (nx != 0) {
                            float d = mx / nx;
                            kernel = __expf( -dot(x,x) / twoSigma2 - d*d / twoRho2 );
                        } else {
                            kernel = (mx != 0)? 0 : 1;
                        }

                        T c = src_(ix + i, iy + j);
                        sum += kernel * c;
                        norm += kernel;
                    }
                }
            } else {
                for (int j = -halfWidth; j <= halfWidth; ++j) {
                    for (int i = -halfWidth; i <= halfWidth; ++i) {
                        float kernel = __expf( -(i*i + j*j) / twoSigma2 );

                        float3 c = src_(ix + i, iy + j);
                        sum += kernel * c;
                        norm += kernel;
                    }
                }
            }
            sum /=  norm;
            return sum;
        }
    };
}


oz::gpu_image oz::hourglass_gauss( const gpu_image& src, const gpu_image& tm,
                                   float sigma, float rho, float precision )
{
    if (sigma <= 0) return src;
    return generate(src.size(), imp_hourglass_gauss<float3>(src, tm, sigma, rho, precision));
}
