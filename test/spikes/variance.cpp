// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "variance.h"
#include <oz/noise.h>
#include <oz/variance.h>
#include <oz/laplacian.h>
#include <oz/hist.h>
#include <oz/color.h>
using namespace oz;


VarianceTest::VarianceTest() {
    new ParamChoice(this, "type", "normal", "uniform|normal|source0", &type);
    new ParamInt   (this, "width",  512, 1, 4096, 1, &w);
    new ParamInt   (this, "height", 512, 1, 4096, 1, &h);
    new ParamDouble(this, "mean",  0.5, 0, 1, 0.1, &mean);
    new ParamDouble(this, "variance",  0.01, 0, 1, 0.005, &variance);
}


void VarianceTest::process() {
    gpu_image src = rgb2gray(gpuInput0());
    if (type == "uniform") {
        src = adjust( noise_uniform(w, h), 2*variance, mean - variance );
    } else if (type == "normal") {
        src = noise_normal(w, h, mean, variance);
    }
    publish("src", src);

    float m, v;
    oz::variance(src, &m, &v);
    qDebug() << "Mean/Variance" << m << v;

    gpu_image L = laplacian(src) * 0.25f;
    publish("laplacian", L * 5);

    oz::variance(L, &m, &v);
    qDebug() << "laplacian" << m << v;

    if (src.is_valid()) {
        publishHistogram("src-H", hist(src, 256, 0, 1));
    }

    if (L.is_valid()) {
        publishHistogram("laplacian-H", hist(L, 256, 0, 1));
    }
}
