// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "filtertest.h"
#include <oz/gauss.h>
#include <oz/median.h>
#include <oz/color.h>
#include <oz/st.h>
#include <oz/stgauss2.h>
#include <oz/dir_gauss.h>
using namespace oz;


FilterTest::FilterTest() {
    new ParamChoice(this, "type", "gauss-3x3", "gauss-3x3|gauss-5x5|median-3x3|stgauss2|dir-gauss", &type);
    new ParamDouble(this, "rho", 2, 0, 20, 0.5, &rho);
    new ParamDouble(this, "sigma", 3, 0, 20, 0.5, &sigma);
    new ParamDouble(this, "alpha", 0, 0, 1000, 0.1, &alpha);
    new ParamDouble(this, "angle", 0, 0, 720, 5, &angle);
}


void FilterTest::process() {
    gpu_image src3 = gpuInput0();
    gpu_image src1 = rgb2gray(src3);

    publish("src-color", src3);
    publish("src-gray", src1);
    if (type == "gauss-3x3") {
        publish("result", gauss_filter_3x3(src1));
    }
    else if (type == "gauss-5x5") {
        publish("result", gauss_filter_5x5(src1));
    }
    else if (type == "median-3x3") {
        publish("result", median_3x3(src1));
    }
    else if (type == "stgauss2") {
        gpu_image st = st_scharr_3x3(src3, 2);
        publish("result", stgauss2_filter(src1, st, 6, 45, false, true, true, 2, 1));
    }
    else if (type == "dir-gauss") {
        gpu_image st = st_scharr_3x3(src3, rho);
        gpu_image tm = st_to_tangent(st);
        publish("result", dir_gauss(src3, tm, sigma, angle, 3));
    }
}
