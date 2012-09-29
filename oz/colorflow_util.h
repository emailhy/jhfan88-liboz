//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// based on code from http://vision.middlebury.edu/flow
//
#pragma once

#include <oz/math_util.h>

namespace oz {

    struct colorflow {
        static const int RY = 15;
        static const int YG = 6;
        static const int GC = 4;
        static const int CB = 11;
        static const int BM = 13;
        static const int MR = 6;
        static const int MAXCOLS = RY + YG + GC + CB + BM + MR;

        static inline __host__ __device__ float3 wheel(int i) {
            if ((i >= 0) && (i < RY)) return make_float3(0, (float)i/RY, 1);
            i -= RY;
            if ((i >= 0) && (i < YG)) return make_float3(0, 1, 1-(float)i/YG);
            i -= YG;
            if ((i >= 0) && (i < GC)) return make_float3((float)i/GC, 1, 0);
            i -= GC;
            if ((i >= 0) && (i < CB)) return make_float3(1, 1-(float)i/CB, 0);
            i -= CB;
            if ((i >= 0) && (i < BM)) return make_float3(1, 0, (float)i/BM);
            i -= BM;
            if ((i >= 0) && (i < MR)) return make_float3(1-(float)i/MR, 0, 1);
            i -= MR;
            return make_float3(0);
        }

        static inline __host__ __device__ float3 value(float2 of) {
            #ifdef __CUDACC__
            if (isnan(of.x) || isnan(of.y)) return make_float3(0);
            #elif _MSC_VER
            if (_isnan(of.x) || _isnan(of.y)) return make_float3(0,0,0);
            #else
            if (isnan(of.x) || isnan(of.y)) return make_float3(0,0,0);
            #endif
            float rad = sqrtf(of.x * of.x + of.y * of.y);
            float a = atan2(-of.y, -of.x) / CUDART_PI_F;
            float fk = (a + 1.0f) / 2.0f * (MAXCOLS-1);
            int k0 = (int)fk;
            int k1 = (k0 + 1) % MAXCOLS;
            float f = fk - k0;
            float3 col0 = wheel(k0);
            float3 col1 = wheel(k1);
            float3 col = (1 - f) * col0 + f * col1;
            if (rad <= 1)
                col = make_float3(1) - rad * (make_float3(1) - col); // increase saturation with radius
            else
                col *= .75; // out of range
            return col;
        }
    };

}
