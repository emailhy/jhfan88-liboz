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

namespace oz {

    class launch_config {
    public:
        launch_config( const gpu_image& dst ) {
            threads_ = dim3(8, 8);
            blocks_ = dim3((dst.w()+threads_.x-1)/threads_.x,
                           (dst.h()+threads_.y-1)/threads_.y );
        }

        dim3 threads() const { return threads_; }
        dim3 blocks() const { return blocks_; }

    private:
        dim3 threads_;
        dim3 blocks_;
    };

}
