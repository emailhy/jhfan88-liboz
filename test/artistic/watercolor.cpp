// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "watercolor.h"
#include <oz/st.h>
#include <oz/gauss.h>
#include <oz/noise.h>
#include <oz/watercolor.h>
using namespace oz;


Watercolor::Watercolor() {
    new ParamDouble(this, "bf_sigma_d", 4, 0,10, 0.5,  &bf_sigma_d);
    new ParamDouble(this, "bf_sigma_r", 15, 0,100, 1,  &bf_sigma_r);
    new ParamInt(this, "bf_N", 4, 1,10, 1, &bf_N);
    new ParamInt(this, "nbins", 16, 1, 100, 1, &nbins);
    new ParamDouble(this, "phi_q", 2, 1, 10, 0.25, &phi_q);
    new ParamDouble(this, "sigma_c", 4, 1, 20,1, &sigma_c);
    new ParamDouble(this, "nalpha", 0.1, 0,1, 0.1, &nalpha);
    new ParamDouble(this, "nbeta", 0.4, 0,1, 0.1, &nbeta);
}


void Watercolor::process() {
    gpu_image src = gpuInput0();
    gpu_image noise = noise_random(src.w(), src.h());
    gpu_image st  = st_scharr_3x3(src,2);
    gpu_image dst = watercolor(
        src,
        st,
        noise,
        bf_sigma_d,
        bf_sigma_r,
        bf_N,
        nbins,
        phi_q,
        sigma_c,
        nalpha,
        nbeta
    );
    publish("$src", src);
    publish("$result", dst);
}
