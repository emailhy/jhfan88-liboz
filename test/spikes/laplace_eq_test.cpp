// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "laplace_eq_test.h"
#include <oz/color.h>
#include <oz/make.h>
#include <oz/shuffle.h>
#include <oz/blend.h>
#include <oz/minmax.h>
#include <oz/test_pattern.h>
#include <oz/resample.h>
#include <oz/blit.h>
#include <oz/colormap.h>
#include <oz/csv.h>
#include <oz/hist.h>
#include <oz/gpu_timer.h>
#include <oz/io.h>
using namespace oz;


LaplaceEqTest::LaplaceEqTest() {
    new ParamChoice(this, "type", "normal", "uniform|normal|source0", &type);
    new ParamChoice(this, "stencil", "stencil20", "stencil4|stencil8|stencil12|stencil20", (int*)&stencil);
    new ParamChoice(this, "upfilt", "nearest", "nearest|linear-fast|linear||cubic-fast|cubic", (int*)&upfilt);
    new ParamInt   (this, "v2", 0, 0, 1000000, 1, &v2);
    new ParamDouble(this, "scale", 1, 0, 1000000, 1, &err_scale);
}


gpu_image LaplaceEqTest::vcycle( const oz::gpu_image b, int level) {
    if ((b.w() <= 2) || (b.h() <= 2)) return b;
    gpu_image tmp = b;

    P.push_back(vstack( tmp.convert(FMT_FLOAT3), shuffle(tmp,3).convert(FMT_FLOAT3),0 ));
    Q.push_back(tmp.convert(FMT_FLOAT3));

    tmp = leq_correct_down(tmp);
    
    tmp = vcycle(tmp, level + 1);

    P.push_back(vstack( tmp.convert(FMT_FLOAT3), shuffle(tmp,3).convert(FMT_FLOAT3),0 ));
    P.push_back(vstack( b.convert(FMT_FLOAT3), shuffle(b,3).convert(FMT_FLOAT3),0 ));

    tmp = leq_correct_up(b, tmp, upfilt);
    
    gpu_image e = leq_residual(tmp, stencil);
    gpu_image eabs = colormap_jet(abs(e.convert(FMT_FLOAT3)), err_scale);

    P.push_back(vstack( tmp.convert(FMT_FLOAT3), shuffle(tmp,3).convert(FMT_FLOAT3), eabs, 0 ));

    for (int k = 0; k < v2; ++k) tmp = leq_jacobi_step(tmp, stencil);

    e = leq_residual(tmp, stencil);
    eabs = colormap_jet(abs(e.convert(FMT_FLOAT3)), err_scale);

    P.push_back(vstack( tmp.convert(FMT_FLOAT3), shuffle(tmp,3).convert(FMT_FLOAT3), eabs,0 ));
    Q.push_back(tmp.convert(FMT_FLOAT3));

    return tmp;
}


void LaplaceEqTest::process() {
    P.clear();
    Q.clear();
    gpu_image hole = gpuInput0().convert(FMT_FLOAT);
    gpu_image fan = resample(test_color_fan(2*hole.w(), 2*hole.h()), hole.w(), hole.h(), RESAMPLE_BSPLINE);
    publish("hole", hole);
    publish("fan", fan);
    gpu_image src = make(blend_intensity(fan, hole, BLEND_MULTIPLY), hole);
    publish("src", src);
    publish("src-w", shuffle(src,3));

    /*{
        gpu_image tmp = src; 
        R.push_back(tmp.convert(FMT_FLOAT3));
        for (int k = 1; k <= 50000; ++k) {
            tmp = leq_jacobi_step(tmp, stencil);
            if ((k == 100) || (k == 1000) || (k == 5000) || (k == 50000)) {
                R.push_back(tmp.convert(FMT_FLOAT3));
            }
        }
        qDebug() << "PERF:" << leq_error(tmp, stencil);
    }
    publishVector("R", R, 10);*/

//     gpu_image cp = make(gpuInput1(),hole);
//     float cperr = leq_error(cp, LEQ_STENCIL_4);
//     qDebug() << "cp error: " << cperr;
// 
//     gpu_image ls = make(gpuInput2(),hole);
//     float lserr = leq_error(ls, LEQ_STENCIL_4);
//     qDebug() << "ls error: " << lserr;
// 
//     gpu_image rs = make(gpuInput3(),hole);
//     float rserr = leq_error(rs, LEQ_STENCIL_4);
//     qDebug() << "ls error: " << rserr;

    /*
    std::vector<double2> R;
    gpu_image s =  src;
    for (int k = 0; k < 100000; ++k) {
        if (k % 100 == 0) {
            float e = leq_error(s);
            R.push_back(make_double2(k, e));
        }
        s = leq_jacobi_step(s);
    }
    csv_write("k", "error", R, "leq_jacobi_err.csv");
    publish("ref", s.convert(FMT_FLOAT3));
    */

    /*
    std::vector<double2> R;
    for (v2 = 0; v2 < 100; ++v2) {
        P.clear();
        gpu_image p1 = vcycle(src, 0);
        double e = leq_error(p1);
        R.push_back(make_double2(v2, e));
        qDebug() << "error: " << v2 << e; 
    }
    csv_write("v2", "error", R, "R3.csv");
    */

    //gpu_image p1 = vcycle(src, 0);
    gpu_timer tt;
    //gpu_image p1 = leq_vcycle(src, v2, stencil, upfilt);
    gpu_image p1 = vcycle(src, 0);
    double t = tt.elapsed_time();
    publishVector("P", P);
    /*{
        gpu_image e = leq_residual(p1, stencil);
        gpu_image eabs = colormap_jet(abs(e.convert(FMT_FLOAT3)), err_scale);
        Q.push_back(eabs);
    }*/
    publishVector("Q", Q, -1320);
    publish("p1", p1.convert(FMT_FLOAT3));
    qDebug() << "p1 error: " << leq_error(p1, stencil) << "time" << t;

//     gpu_image e = leq_residual(p1, stencil);
//     publish("e", e.convert(FMT_FLOAT3) * err_scale);
//     gpu_image e_abs = abs(e);
//     publish("e_abs", colormap_jet(e_abs, err_scale));
// 
//     float e_min, e_max;
//     minmax(e_abs, &e_min, &e_max);
//     std::vector<int> e_H = hist(e_abs, 256, e_min, 0.001f);
//     qDebug() << "p1 error: " << leq_error(p1, stencil) << "min" << e_min << "max" << e_max; 
//     publishHistogram("e_H", e_H, 1000);


//     gpu_image d1 = p1.convert(FMT_UCHAR4).convert(FMT_FLOAT4);
//     qDebug() << "d1 error: " << leq_error(d1, stencil); 
//     gpu_image d = leq_residual(d1, stencil);
//     publish("d", d.convert(FMT_FLOAT3) * err_scale);
//     publish("d_abs", colormap_jet(abs(d), err_scale));
}
