// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "convpyr_test.h"
#include <oz/convpyr.h>
#include <oz/color.h>
#include <oz/make.h>
#include <oz/shuffle.h>
#include <oz/blend.h>
#include <oz/test_pattern.h>
#include <oz/resample.h>
#include <oz/blit.h>
#include <oz/colormap.h>
#include <oz/pad.h>
#include <oz/laplace_eq.h>
#include <oz/gpu_timer.h>
using namespace oz;


ConvPyrTest::ConvPyrTest() {
}


void ConvPyrTest::process() {
    gpu_image hole = gpuInput0().convert(FMT_FLOAT);
    gpu_image fan = resample(test_color_fan(2*hole.w(), 2*hole.h()), hole.w(), hole.h(), RESAMPLE_BSPLINE);
    publish("hole", hole);
    publish("fan", fan);
    //gpu_image src = make(blend_intensity(fan, hole, BLEND_MULTIPLY), hole);
    gpu_image src = make(fan, hole);
    publish("src", src);
    publish("src-w", shuffle(src,3));


    gpu_timer tt;
    gpu_image a = convpyr_boundary(src);
    double t = tt.elapsed_time();
    qDebug() << "cp err" << leq_error(make(a, hole), LEQ_STENCIL_12) << "time" << t;

    publish("a.xyz", a.convert(FMT_FLOAT3));
    publish("a.w", shuffle(a, 3));
}
