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
#include <oz/threshold.h>
#include <oz/hist.h>
#include <oz/transform.h>


namespace oz {

    struct op_threshold : public unary_function<float,float> {
        float t_;
        op_threshold(float t) : t_(t) {}
        inline __device__ float operator()(float v) const {
            return (v <= t_)? 0 : 1;
        }
    };

    gpu_image threshold( const gpu_image& src, float t ) {
        return transform(src, op_threshold(t));
    }


    gpu_image otsu( const gpu_image& src ) {
        std::vector<int> H = hist(src, 256);

        float sum = 0;
        for (int k = 0; k < 256; ++k) sum += k * H[k];

        float sumB = 0;
        int pB = 0;
        int pF = 0;

        float var_max = 0;
        int var_idx = 0;

        for (int k = 0; k < 256; ++k) {
           pB += H[k];
           if (pB == 0) continue;

           pF = src.N() - pB;
           if (pF == 0) break;

           sumB += k * H[k];

           float meanB = sumB / pB;
           float meanF = (sum - sumB) / pF;

           float var_between = pB * pF * (meanB - meanF) * (meanB - meanF);
           if (var_between > var_max) {
              var_max = var_between;
              var_idx = k;
           }
        }

        return threshold(src, var_idx/255.0f);
    }

}
