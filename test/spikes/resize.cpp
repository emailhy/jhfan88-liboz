// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "resize.h"
#include <oz/resize.h>
#include <oz/color.h>
#include <oz/gpu_cache.h>
using namespace oz;


ResizeTest::ResizeTest() {
    new ParamChoice(this, "mode", 0, "nearest|fast_bilinear|bilinear|fast_bicubic|bicubic|catrom", &mode);
    new ParamInt   (this, "width",  512, 1, 4096, 1, &w);
    new ParamInt   (this, "height", 512, 1, 4096, 1, &h);
}


void ResizeTest::process() {
    gpu_cache_clear();
    gpu_image src3 = gpuInput0();
    gpu_image src1 = rgb2gray(src3);
    gpu_image dst1 = resize(src1, w, h, (resize_mode_t)mode);
    gpu_image dst3 = resize(src3, w, h, (resize_mode_t)mode);
    publish("src3", src3);
    publish("src1", src1);
    publish("dst1", dst1);
    publish("dst3", dst3);
}
