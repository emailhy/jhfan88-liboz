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
#include <oz/msst.h>
#include <oz/st_util.h>
#include <oz/transform.h>
#include <vector>

namespace oz {

    struct StMoaMerge : public binary_function<float3,float3,float3> {
        float epsilon_;
        moa_t moa_;

        StMoaMerge( float epsilon, moa_t moa )
            : epsilon_(epsilon), moa_(moa) {}

        inline __device__ float3 operator()(float3 a, float3 b) const {

            float wa = st2moa(a, moa_);
            float wb = st2moa(b, moa_);

            float w = wa + wb;
            if (w < epsilon_) {
                return 0.5f * (a + b);
            } else {
                return a * wa/(wa+wb) + b * wb/(wa+wb);
            }
        }
    };

    gpu_image st_moa_merge( const gpu_image& stA, const gpu_image& stB,
                            float epsilon, moa_t moa )
    {
        return transform(stA, stB, StMoaMerge(epsilon, moa));
    }


    gpu_image st_multi_scale( const gpu_image& src, int max_depth, float rho,
                              resample_mode_t resample_mode, float epsilon, moa_t moa )
    {
        std::vector<gpu_image> P;
        {
            gpu_image cur = src;
            while ((int)P.size() < max_depth) {
                P.push_back(cur);
                if ((cur.w() <= 1) || (cur.h() <= 1)) break;
                cur = resample(cur, (cur.w()+1)/2, (cur.h()+1)/2, resample_mode);
            }
        }

        gpu_image st_cur;
        gpu_image st_prev;
        for (int k = (int)P.size() - 1; k >= 0; --k) {
            st_cur  = st_scharr_3x3(P[k], rho);
            if (k < (int)P.size() - 1) {
                st_prev = resample(st_prev, P[k].w(), P[k].h(), RESAMPLE_TRIANGLE);
                st_cur = st_moa_merge(st_cur, st_prev, epsilon, moa);
            }
            st_prev = st_cur;
        }
        return st_cur;
    }

}
