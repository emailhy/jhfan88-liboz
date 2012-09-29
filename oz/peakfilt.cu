//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// based on a MATLAB implementation by Thomas Pock 
// Copyright 2011 Adobe Systems Incorporated 
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
#include <oz/peakfilt.h>
#include <oz/median.h>
#include <oz/variance.h>
#include <oz/generate.h>


namespace {
    struct PeakFlt3x3 : public oz::generator<float> {
        oz::gpu_plm2<float> src_;
        oz::gpu_plm2<float> med_;
        oz::gpu_plm2<float> d_;
        float mean_;
  
        PeakFlt3x3( const oz::gpu_image& src, const oz::gpu_image& med, 
                    const oz::gpu_image& d, float mean )
            : src_(src), med_(med), d_(d), mean_(mean) {}
        
        inline __device__ float operator()( int ix, int iy ) const {
            return (d_(ix,iy) <= mean_)? src_(ix,iy) : med_(ix,iy);
        }
    };
}


oz::gpu_image oz::peakfilt_3x3( const gpu_image& src ) {
    gpu_image med = median_3x3(src);
    gpu_image d = abs_diff(med, src);
    float m = mean(abs(src));
    return generate(src.size(), PeakFlt3x3(src, med, d, m));
}
