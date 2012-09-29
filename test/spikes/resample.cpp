// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "resample.h"
#include <oz/resample.h>
#include <oz/color.h>
#include <oz/shuffle.h>
#include <oz/test_pattern.h>
#include <oz/gpu_cache.h>
using namespace oz;


ResampleTest::ResampleTest() {
    ParamGroup *g;

    g = new ParamGroup(this, "input");
    new ParamChoice(g, "image_type", "image", "image|zone-plate", &image_type);
    new ParamInt(g, "zwidth", 512, 1, 4096, 1, &zwidth);
    new ParamInt(g, "zheight", 512, 1, 4096, 1, &zheight);
    new ParamDouble(g, "g0", 1, 0, 100, 0.01, &g0);
    new ParamDouble(g, "km", 2, 0, 4096, 0.01, &km);
    new ParamDouble(g, "rm", 240, 0, 4096, 0.25, &rm);
    new ParamDouble(g, "w",  8, 0, 4096, 0.25, &w);
    new ParamBool  (g, "inverted", false, &inverted);
    new ParamBool  (g, "sRGB", false, &sRGB);

    g = new ParamGroup(this, "resample");
    new ParamChoice(g, "mode", 0, "box|triangle|bell|quadratic|quadratic-approx|quadratic-mix|bspline|lanczos2|lanczos3|blackman|cubic|catrom|mitchell|gaussian|kaiser", &mode);
    new ParamInt (g, "rwidth",  512, 1, 4096, 1, &rwidth);
    new ParamInt (g, "rheight", 512, 1, 4096, 1, &rheight);
}


void ResampleTest::process() {
    gpu_cache_clear();

    gpu_image src;
    if (image_type == "zone-plate") {
        src = shuffle(test_zoneplate(zwidth, zheight, g0, km, rm, w, inverted), 0);
    } else {
        src = gpuInput0();
    }

    gpu_image dst = resample(src, rwidth, rheight, (resample_mode_t)mode);
    publish("src", sRGB? linear2srgb(src) : src);
    publish("dst", sRGB? linear2srgb(dst) : dst);
    /*
    gpu_image<float> src1 = gpu_rgb2gray(src4);
    gpu_image<float> dst1 = gpu_resample(src1, w, h, (gpu_resample_mode_t)mode);

    if ((w == (src1.w()+1)/2) && (h == (src1.h()+1)/2)) {
        gpu_image<float> pd = gpu_pyrdown_gauss5x5(src1);
        gpu_image<float4> diff = gpu_color_diff(pd, dst1, 0.1f);
        publish("pd", pd);
        publish("pd_diff", diff);

        gpu_image<float> half = gpu_resize_half(src1);
        diff = gpu_color_diff(half, dst1, 0.1f);
        publish("half", half);
        publish("half_diff", diff);
    }

    //gpu_image<float4> dst4 = gpu_resample(src4, w, h, (gpu_resample_mode_t)mode);
    publish("src4", src4);
    publish("src1", src1);
    publish("dst1", dst1);

    {
        gpu_image<float> bi = gpu_resize(src1, w, h, GPU_RESIZE_BILINEAR);
        gpu_image<float4> diff = gpu_color_diff(bi, dst1, 0.1f);
        publish("dst1-bi", bi);
        publish("dst1-bi-diff", diff);
    }

    {
        gpu_image<float> c = gpu_resize(src1, w, h, GPU_RESIZE_CATROM);
        gpu_image<float4> diff = gpu_color_diff(c, dst1, 0.1f);
        publish("dst1-c", c);
        publish("dst1-c-diff", diff);
    }

    //publish("dst4", dst4);
    */
}
