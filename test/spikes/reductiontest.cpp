// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "reductiontest.h"
#include <oz/color.h>
#include <oz/gpu_timer.h>
#include <oz/minmax.h>
#include <npp.h>
using namespace oz;


void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata);
void gpu_minmax2(const gpu_image& src, float *pmin, float *pmax);


float reduceCPU(float *data, int size) {
    float sum = data[0];
    float c = (float)0.0;
    for (int i = 1; i < size; i++)
    {
        float y = data[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}


ReductionTest::ReductionTest() {
    new ParamInt   (this, "threads", 256, 1, 512, 1, &threads);
    new ParamInt   (this, "blocks", 64, 1, 512, 1, &blocks);
}


void ReductionTest::process() {
    gpu_image src = gpuInput0();
    gpu_image gray = rgb2gray(src);
    gpu_image r(gray.w(), gray.h(), FMT_FLOAT);

    cpu_image cgray = gray.cpu();
    float sum = reduceCPU(cgray.ptr<float>(), cgray.w()*cgray.h());
    qDebug() << "Sum(CPU):" << sum;
    {
        const unsigned N = cgray.N();
        float pmin = *cgray.ptr<float>();
        float pmax = pmin;
        for (unsigned i = 1; i < N; ++i) {
            pmin = fminf(pmin, *(cgray.ptr<float>() + i));
            pmax = fmaxf(pmax, *(cgray.ptr<float>() + i));
        }
        qDebug() << "min/max(CPU):" << pmin << pmax;
    }


    for (int k = 0; k < 10; ++k) {
        gpu_timer tt;
        reduce(gray.w()*gray.h(), threads, blocks, gray.ptr<float>(), r.ptr<float>());
        float t = tt.elapsed_time();
        qDebug() << "Time #0: " << t << "T:" << 1.0e-6 * ((double)4*gray.w()*gray.h()) / t;
    }
    cpu_image cr = r.cpu();
    sum = 0;
    for (int i =0; i < blocks; ++i) sum += cr.at<float>(i,0);
    qDebug() << "Sum  #0 (GPU):" << sum;

    float pmin = 0;
    float pmax = 0;
    for (int k = 0; k < 10; ++k) {
        gpu_timer tt;
        minmax(gray, &pmin, &pmax);
        float t = tt.elapsed_time();
        qDebug() << "Time #1: " << t << "T:" << 1.0e-6 * ((double)4*gray.w()*gray.h()) / t;
    }
    qDebug() << "#1" << "Min:"<< pmin << "Max:" << pmax;

    for (int k = 0; k < 10; ++k) {
        gpu_timer tt;
        gpu_minmax2(gray, &pmin, &pmax);
        float t = tt.elapsed_time();
        qDebug() << "Time #1: " << t << "T:" << 1.0e-6 * ((double)4*gray.w()*gray.h()) / t;
    }
    qDebug() << "#2" << "Min:"<< pmin << "Max:" << pmax;

    /*{
        gpu_image<unsigned char> gray8 = gpu_32f_to_8u(gray);
        NppiSize sizeROI = { src.w(), src.h() };
        Npp8u pMin;
        Npp8u pMax;
        gpu_timer tt;
        nppiMinMax_8u_C1R((const Npp8u*)gray8.ptr(), gray8.pitch(), sizeROI, &pMin, &pMax);
        float t = tt.elapsed_time();
        qDebug() << "Time #2: " << t << "T:" << 1.0e-6 * ((double)gray.w()*gray.h()) / t;
        qDebug() << pMin << pMax;
    }
    publish("$gray", gray);
    */
}
