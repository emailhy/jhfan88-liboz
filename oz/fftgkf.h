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
#pragma once

#include <oz/gpu_image.h>
#include <cufft.h>

namespace oz {

    struct OZAPI fftgkf_t {
        fftgkf_t( unsigned w, unsigned h, float sigma_s, float sigma_r );
        ~fftgkf_t();
        gpu_image operator()( const gpu_image& src, float threshold, float q ) const;

        unsigned fw_;
        unsigned fh_;
        gpu_image krnl_[8];
        cufftComplex* kernel_[8];
        cufftComplex *spec_, *tspec_;
        cufftReal *data_;
        cufftReal *res_;
        cufftHandle planf_, plani_;
    };


    OZAPI gpu_image gkf_simple( const gpu_image& src, float sigma_s, float sigma_r,
                                float threshold, float q );

}

