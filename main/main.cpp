//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2011-2012 Computer Graphics Systems Group at the
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
#include <oz/gpu_image.h>
#include <oz/gpu_timer.h>
#include <oz/io.h>
#include <oz/gauss.h>
#include <oz/qimage.h>
#include <oz/exception.h>
#include <iostream>
#include <sstream>
using namespace oz;


int main(int argc, char **argv) {
    try {
        gpu_image I = imread("test.tga").convert(FMT_FLOAT3);
        gpu_timer gt;
        for (int i = 0; i < 10; ++i) {
            I = gauss_filter(I, 5);
        }
        double t = gt.elapsed_time();
        imwrite(I, "test-out.tga");
    }
    catch (std::exception& e) {
        std::cerr << "***ERROR***" << std::endl;
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
