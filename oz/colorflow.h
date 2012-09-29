//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// based on code from http://vision.middlebury.edu/flow
//
#pragma once

#include <oz/gpu_image.h>

namespace oz {

    OZAPI gpu_image colorflow( const gpu_image& src, float radius_max=0, bool robust=true );
    OZAPI gpu_image magflow( const gpu_image& f );

}
