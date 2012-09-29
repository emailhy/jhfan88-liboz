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
#include <oz/laplace_eq.h>
#include <oz/color.h>
#include <oz/gauss.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>
#include <oz/gpu_plm2.h>
#include <oz/tex2d_util.h>
#include <oz/norm.h>


namespace oz {

    struct LEqJacobiStep : public generator<float4> {
        gpu_sampler<float4,0> src_;
        leq_stencil_t stencil_;

        LEqJacobiStep( const gpu_image& src, leq_stencil_t stencil)
            : src_(src), stencil_(stencil) {}

        inline __device__ float4 operator()( int ix, int iy )const {
            float4 o = src_(ix, iy);
            if (o.w < 1) {
                switch (stencil_) {
                    case LEQ_STENCIL_4:
                        o = make_float4( 0.25f * (
                            make_float3(src_(ix,   iy+1)) +
                            make_float3(src_(ix-1, iy  )) +
                            make_float3(src_(ix+1, iy  )) +
                            make_float3(src_(ix  , iy-1))), 0);
                        break;
                    case LEQ_STENCIL_8:
                        o = make_float4( 0.125f * (
                            make_float3(src_(ix+1, iy+1)) +
                            make_float3(src_(ix,   iy+1)) +
                            make_float3(src_(ix-1, iy+1)) +
                            make_float3(src_(ix-1, iy  )) +
                            make_float3(src_(ix+1, iy  )) +
                            make_float3(src_(ix+1, iy-1)) +
                            make_float3(src_(ix  , iy-1)) +
                            make_float3(src_(ix-1, iy-1))), 0);
                        break;
                    case LEQ_STENCIL_12:
                        o = make_float4( 1.0f / 12.0f * (
                            1 * make_float3(src_(ix+1, iy+1)) +
                            2 * make_float3(src_(ix,   iy+1)) +
                            1 * make_float3(src_(ix-1, iy+1)) +
                            2 * make_float3(src_(ix-1, iy  )) +
                            2 * make_float3(src_(ix+1, iy  )) +
                            1 * make_float3(src_(ix+1, iy-1)) +
                            2 * make_float3(src_(ix  , iy-1)) +
                            1 * make_float3(src_(ix-1, iy-1))), 0);
                        break;
                    case LEQ_STENCIL_20:
                        o = make_float4( 1.0f / 20.0f * (
                            1 * make_float3(src_(ix+1, iy+1)) +
                            4 * make_float3(src_(ix,   iy+1)) +
                            1 * make_float3(src_(ix-1, iy+1)) +
                            4 * make_float3(src_(ix-1, iy  )) +
                            4 * make_float3(src_(ix+1, iy  )) +
                            1 * make_float3(src_(ix+1, iy-1)) +
                            4 * make_float3(src_(ix  , iy-1)) +
                            1 * make_float3(src_(ix-1, iy-1))), 0);
                        break;
                }
            }
            return o;
        }
    };

    gpu_image leq_jacobi_step( const gpu_image& src, leq_stencil_t stencil ) {
        return generate(src.size(), LEqJacobiStep(src, stencil));
    }


    struct LEqCorrectDown : public generator<float4> {
        gpu_sampler<float4,0> src_;

        LEqCorrectDown( const gpu_image& src ) : src_(src) {}

        inline __device__ float4 operator()( int ix, int iy ) const {
            int i = 2*ix;
            int j = 2*iy;
            float4 sum = make_float4(0);
            float4 c;
            c = src_(i,   j  ); if (c.w > 0) { sum += c; }
            c = src_(i+1, j  ); if (c.w > 0) { sum += c; }
            c = src_(i  , j+1); if (c.w > 0) { sum += c; }
            c = src_(i+1, j+1); if (c.w > 0) { sum += c; }
            if (sum.w > 0) sum /= sum.w;
            return sum;
        }
    };


    gpu_image leq_correct_down( const gpu_image& src) {
        gpu_image r = generate((src.w()+1)/2, (src.h()+1)/2, LEqCorrectDown(src));
        return r;
    }


    struct LEqCorrectUp : public generator<float4> {
        gpu_plm2<float4> src0_;
        gpu_sampler<float4,0> src1_;
        int upfilt_;

        LEqCorrectUp( const gpu_image& src0, const gpu_image& src1, leq_upfilt_t upfilt )
            : src0_(src0),
              src1_(src1, ((upfilt==LEQ_UPFILT_FAST_BILINEAR) || (upfilt==LEQ_UPFILT_FAST_BICUBIC))? cudaFilterModeLinear : cudaFilterModePoint),
              upfilt_(upfilt) {}

        inline __device__ float4 operator()( int ix, int iy )const {
            float4 c = src0_(ix, iy);
            if (c.w < 1) {
                float2 uv = make_float2(0.5f * (ix + 0.5f), 0.5f * (iy + 0.5f));
                switch (upfilt_) {
                    case LEQ_UPFILT_NEAREST:
                        c = make_float4(make_float3(src1_(ix/2, iy/2)), 0);
                        break;
                    case LEQ_UPFILT_FAST_BILINEAR:
                        c = make_float4(make_float3(src1_(uv.x, uv.y)), 0);
                        break;
                    case LEQ_UPFILT_BILINEAR:
                        c = make_float4(make_float3(tex2DBilinear(src1_.texref(), uv.x, uv.y)), 0);
                        break;
                    case LEQ_UPFILT_FAST_BICUBIC:
                        c = make_float4(make_float3(tex2DFastBicubic(src1_.texref(), uv.x, uv.y)), 0);
                        break;
                    case LEQ_UPFILT_BICUBIC:
                        c = make_float4(make_float3(tex2DBicubic(src1_.texref(), uv.x, uv.y)), 0);
                        break;
                }
            }
            return c;
        }
    };

    gpu_image leq_correct_up( const gpu_image& src0, const gpu_image& src1, leq_upfilt_t upfilt ) {
        gpu_image r = generate(src0.size(), LEqCorrectUp(src0, src1, upfilt));
        return r;
    }


    struct LEqResidual : public oz::generator<float3> {
        gpu_sampler<float4,0> src_;
        leq_stencil_t stencil_;

        LEqResidual( const gpu_image& src, leq_stencil_t stencil )
            : src_(src), stencil_(stencil) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float4 c = src_(ix, iy);
            if (c.w < 1) {
                switch (stencil_) {
                    case LEQ_STENCIL_4:
                        return (make_float3(src_(ix  , iy+1)) +
                                make_float3(src_(ix+1, iy  )) +
                                make_float3(src_(ix-1, iy  )) +
                                make_float3(src_(ix  , iy-1)) - 4 * make_float3(c)) / 4;
                    case LEQ_STENCIL_8:
                        return (make_float3(src_(ix+1, iy+1)) +
                                make_float3(src_(ix  , iy+1)) +
                                make_float3(src_(ix-1, iy+1)) +
                                make_float3(src_(ix+1, iy  )) +
                                make_float3(src_(ix-1, iy  )) +
                                make_float3(src_(ix+1, iy-1)) +
                                make_float3(src_(ix  , iy-1)) +
                                make_float3(src_(ix-1, iy-1)) - 8 * make_float3(c)) / 8;
                    case LEQ_STENCIL_12:
                        return (1*make_float3(src_(ix+1, iy+1)) +
                                2*make_float3(src_(ix  , iy+1)) +
                                1*make_float3(src_(ix-1, iy+1)) +
                                2*make_float3(src_(ix+1, iy  )) +
                                2*make_float3(src_(ix-1, iy  )) +
                                1*make_float3(src_(ix+1, iy-1)) +
                                2*make_float3(src_(ix  , iy-1)) +
                                1*make_float3(src_(ix-1, iy-1)) - 12 * make_float3(c)) / 12;
                    case LEQ_STENCIL_20:
                        return (1*make_float3(src_(ix+1, iy+1)) +
                                4*make_float3(src_(ix  , iy+1)) +
                                1*make_float3(src_(ix-1, iy+1)) +
                                4*make_float3(src_(ix+1, iy  )) +
                                4*make_float3(src_(ix-1, iy  )) +
                                1*make_float3(src_(ix+1, iy-1)) +
                                4*make_float3(src_(ix  , iy-1)) +
                                1*make_float3(src_(ix-1, iy-1)) - 20 * make_float3(c)) / 20;
                }
            }
            return make_float3(0);
        }
    };

    gpu_image leq_residual( const gpu_image& src, leq_stencil_t stencil ) {
        return generate(src.size(), LEqResidual(src, stencil));
    }


    float leq_error( const gpu_image& src, leq_stencil_t stencil ) {
        return sqrtf(sum(abs2(leq_residual(src, stencil))));
    }


    gpu_image leq_vcycle( const gpu_image& b, int v2, leq_stencil_t stencil, leq_upfilt_t upfilt ) {
        if ((b.w() <= 2) || (b.h() <= 2)) return b;
        gpu_image tmp = b;
        tmp = leq_correct_down(tmp);
        tmp = leq_vcycle(tmp, v2, stencil, upfilt);
        tmp = leq_correct_up(b, tmp, upfilt);
        for (int k = 0; k < v2; ++k) tmp = leq_jacobi_step(tmp, stencil);
        return tmp;
    }

}
