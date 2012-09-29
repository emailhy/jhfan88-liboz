// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "ffttest.h"
#include <oz/color.h>
//#include <oz/colormap.h>
#include <oz/minmax.h>
#include <oz/hist.h>
#include <oz/fft.h>
#include <oz/variance.h>
using namespace oz;


FFTTest::FFTTest() {
}


void FFTTest::process() {
    gpu_image src = rgb2gray(gpuInput0());
    publish("src", src);

    gpu_image F = fft2(src);
    cpu_image cF = F.cpu();

    qDebug() << "F[0][0]" << cF.at<float2>(0,0).x << cF.at<float2>(0,0).y;
    float m = mean(src);
    qDebug() << "mean" << m;
    qDebug() << "F/" << cF.at<float2>(0,0).x / (src.N());

    gpu_image L = log_abs(F);
    std::vector<int> H = hist(L, 1024, -1, 10);
    publishHistogram("H", H);

    L = abs(F);
    gpu_image P = fftshift(L * (6.0f / (src.N())));
    publish("P", P);
    //publish("P-color", gpu_colormap_jet(P));
}
