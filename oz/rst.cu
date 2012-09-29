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
#include <oz/rst.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {
    struct RstSmooth : public oz::generator<float3> {
        gpu_sampler<float3,0> st_;
        float rho_;
        float m_;
        float precision_;

        RstSmooth( const oz::gpu_image& st, float rho, float m, float precision )
            : st_(st), rho_(rho), m_(m), precision_(precision) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float twoRho2 = 2.0f * rho_ * rho_;
            float twoM2 = 2.0f * m_ * m_;
            int halfWidth = int(ceilf( precision_ * rho_ ));

            float3 sum = make_float3(0);
            float w = 0;

            for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                    float d = length(make_float2(i,j));
                    float3 gg = st_(ix + i, iy + j);
                    float K_rho = __expf( -d*d / twoRho2 );
                    sum += K_rho * gg;
                    w += K_rho;
                }
            }
            sum /=  w;

            for (int k = 0; k < 5; ++k) {
                float2 v = oz::st2gradient(sum);
                sum = make_float3(0);
                w = 0;

                for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                    for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                        float d = length(make_float2(i,j));

                        float3 gg = st_(ix + i, iy + j);
                        float x = gg.x + gg.y - v.x*v.x*gg.x - v.x*v.y*gg.z - v.y*v.y*gg.y;

                        float K_rho = __expf( -d*d / twoRho2 );
                        float K_m = __expf( -x*x / twoM2 );
                        float k = K_rho * K_m;

                        sum += k * gg;
                        w += k;
                    }
                }
                sum /=  w;
            }

            return sum;
        }
    };
}


oz::gpu_image oz::rst_scharr( const gpu_image& src, float rho, float m ) {
    gpu_image st = st_scharr_3x3(src);
    return generate(st.size(), RstSmooth(st, rho, m, 2));
}
