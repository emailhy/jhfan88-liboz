// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "otsu.h"
#include <oz/color.h>
#include <oz/threshold.h>
using namespace oz;


OtsuTest::OtsuTest() {
}


void OtsuTest::process() {
    gpu_image src = gpuInput0();
    gpu_image gray = rgb2gray(src);

    publish("gray", gray);
    publish("T", otsu(gray));
}
