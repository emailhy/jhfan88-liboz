// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "noise.h"
#include <oz/noise.h>
#include <oz/hist.h>
#include <oz/gpu_timer.h>
#include <oz/minmax.h>
#include <oz/color.h>
#include <oz/cpu_timer.h>
using namespace oz;


NoiseTest::NoiseTest() {
    new ParamChoice(this, "type", "uniform", "uniform|normal|linear_noise|rand|gaussian|salt_and_pepper|speckle", &type);
    new ParamInt   (this, "width",  512, 1, 4096, 1, &w);
    new ParamInt   (this, "height", 512, 1, 4096, 1, &h);
    new ParamDouble(this, "mean",  0.5, 0, 1, 0.1, &mean);
    new ParamDouble(this, "variance",  0.01, 0, 1, 0.005, &variance);
    new ParamDouble(this, "scale",  1, 0, 10, 0.05, &scale);
    new ParamDouble(this, "density",  0.05, 0, 1, 0.01, &density);
    new ParamBool  (this, "normalize", false, &normalize);
}


void NoiseTest::process() {
    gpu_image dst;
    if (type == "uniform") {
        gpu_timer gt;
        dst = noise_uniform(w, h);
        float t = gt.elapsed_time();
        qDebug() << "uniform" << t;
        publish("result", dst);
    }
    else if (type == "normal") {
        gpu_timer gt;
        dst = noise_normal(w, h, mean, variance);
        float t = gt.elapsed_time();
        qDebug() << "normal" << t;
        publish("result", dst);
    }
    else if (type == "linear_noise") {
        gpu_timer gt;
        dst = noise_fast(w, h, scale);
        float t = gt.elapsed_time();
        qDebug() << "fast" << t;
        publish("result", dst);
    }
    else if (type == "rand") {
        cpu_timer ct;
        dst = noise_random(w, h);
        float t = ct.elapsed_time();
        qDebug() << "rand" << t;
        publish("result", dst);
    }
    else if (type == "gaussian") {
        gpu_image src3 = gpuInput0();
        gpu_image dst3 = add_gaussian_noise(src3, mean, variance);
        publish("result", dst3);

        gpu_image src1 = rgb2gray(src3);
        gpu_image dst1 = add_gaussian_noise(src1, mean, variance);
        publish("result gray", dst1);
    }
    else if (type == "salt_and_pepper") {
        gpu_image src3 = gpuInput0();
        gpu_image dst3 = add_salt_and_pepper_noise(src3, density);
        publish("result", dst3);

        gpu_image src1 = rgb2gray(src3);
        gpu_image dst1 = add_salt_and_pepper_noise(src1, density);
        publish("result gray", dst1);
    }
    else if (type == "speckle") {
        gpu_image src3 = gpuInput0();
        gpu_image dst3 = add_speckle_noise(src3, density);
        publish("result", dst3);

        gpu_image src1 = rgb2gray(src3);
        gpu_image dst1 = add_speckle_noise(src1, variance);
        publish("result", dst1);
    }

    if (dst.is_valid()) {
        publishHistogram("histogram", hist(dst, 256, 0, 1));
    }
}
