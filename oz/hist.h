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
#include <vector>

namespace oz {

    OZAPI std::vector<int> hist( const gpu_image& src );
    OZAPI std::vector<int> hist( const gpu_image& src, int nbins, float pmin=0, float pmax=1 );
    OZAPI std::vector<int> hist_join( const std::vector<int>& H0, const std::vector<int>& H1 );
    OZAPI void hist_insert( std::vector<int>& H, float value, float pmin=0, float pmax=1);
    OZAPI std::vector<float> hist_to_pdf( const std::vector<int>& H, float a, float b );
    OZAPI float pdf_sgnf( const std::vector<float>& pdf, float a, float b, float s );
    OZAPI std::vector<float> pdf_to_cdf( const std::vector<float>& pdf, float a, float b );
    OZAPI gpu_image hist_eq( const gpu_image& src );
    OZAPI gpu_image hist_auto_levels( const gpu_image& src, float threshold = 0.1 );
    OZAPI std::vector<float> pdf( const gpu_image& src, int nbins, float pmin=0, float pmax=1 );

}
