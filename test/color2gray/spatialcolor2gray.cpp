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
#include "spatialcolor2gray.h"
#include <oz/spatial_color2gray.h>
#include <oz/color.h>
#include <oz/shuffle.h>
using namespace oz;


SpatialColor2Gray::SpatialColor2Gray() {
    new ParamInt   (this, "radius", 2, 1, 20, 1, &radius);
    new ParamDouble(this, "K", 1, 0, 100, 1, &K);
    new ParamDouble(this, "B1", 15, 0, 100, 1, &B1);
    new ParamDouble(this, "B2", 40, 0, 100, 1, &B2);
}


void SpatialColor2Gray::process() {
    gpu_image src = gpuInput0();
    publish("src", src);
    gpu_image lab = rgb2lab(src);
    gpu_image G = spatial_color2gray(lab, radius, K, B1, B2);
    publish("G", G/100.0f);
    publish("L", shuffle(lab,0)/100.0f);
}
