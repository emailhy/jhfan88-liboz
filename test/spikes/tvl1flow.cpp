// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "tvl1flow.h"
#include <oz/color.h>
#include <oz/colorflow.h>
#include <oz/gauss.h>
#include <oz/tvl1flow.h>
#include <oz/variance.h>
using namespace oz;


TVL1Flow::TVL1Flow() {
    ParamGroup *g; 
    
    g = new ParamGroup(this, "optical flow");
    new ParamDouble(g, "pre_smooth", 1, 0, 5, 0.1, &pre_smooth);
    new ParamDouble(g, "pyr_scale", 0.8, 0.5, 0.98, 0.1, &pyr_scale);
    new ParamInt   (g, "warps", 1, 1, 10, 1, &warps);
    new ParamInt   (g, "maxits", 50, 1, 100, 1, &maxits);
    new ParamDouble(g, "lambda", 50, 0, 1000, 1, &lambda);
    new ParamDouble(g, "max_radius", 1, 0,10, 0.1, &max_radius);
}



void TVL1Flow::process() {
    gpu_image c0 = gpuInput0(-1);
    gpu_image c1 = gpuInput0(0);
    gpu_image s0 = gauss_filter_xy(rgb2gray(c0), pre_smooth);
    gpu_image s1 = gauss_filter_xy(rgb2gray(c1), pre_smooth);
    publish("s0", s0);
    publish("s1", s1);

    gpu_image f01 = tvl1flow(s0, s1, pyr_scale, warps, maxits, lambda);
    gpu_image f10 = tvl1flow(s1, s0, pyr_scale, warps, maxits, lambda);
    publish("f01", colorflow(f01, max_radius));
    publish("f10", colorflow(-f10, max_radius));
}
