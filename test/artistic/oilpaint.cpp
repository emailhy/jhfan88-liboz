// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "oilpaint.h"
#include <oz/st.h>
#include <oz/gauss.h>
#include <oz/noise.h>
#include <oz/oily2.h>
using namespace oz;


OilPaint::OilPaint() {
    new ParamDouble(this, "st_sigma",    2,  0,  20,   1,  &m_st_sigma);
    new ParamDouble(this, "flow_blur",    8,  0,  20,   1,  &m_flow_blur);
    new ParamDouble(this, "bump_scale", 5,   0,  10,   1,  &m_bump_scale);
    new ParamDouble(this, "phong_specular", 2,   0,  10,   1,  &m_phong_specular);
    new ParamDouble(this, "phong_shininess", 7,   0,  100,   1,  &m_phong_shininess);
}


void OilPaint::process() {
    gpu_image src = gpuInput0();
    gpu_image noise = noise_random(src.w(), src.h());
    gpu_image st  = st_scharr_3x3(src, m_st_sigma);
    gpu_image dst = oily2(src, st, noise, m_flow_blur, m_bump_scale, m_phong_specular, m_phong_shininess);
    publish("$src", src);
    publish("$result", dst);
}
