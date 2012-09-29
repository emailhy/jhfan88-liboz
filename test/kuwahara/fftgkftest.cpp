//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
#include "fftgkftest.h"
#include <oz/color.h>
#include <oz/fftgkf.h>
#include <oz/gkf_kernel.h>
#include <oz/gkf_opt2.h>
#include <oz/gpu_timer.h>
#include <oz/gpu_cache.h>
#include <oz/progress.h>
#include <oz/csv.h>
using namespace oz;


FFTGkfTest::FFTGkfTest() {
    new ParamInt   (this, "N", 8, 2, 8, 1, &N_);
    new ParamInt   (this, "k", 0, 0, 7, 1, &k_);
    new ParamDouble(this, "radius", 3, 0, 100, 1, &radius_);
    new ParamDouble(this, "precision", 2.5, 1, 5, 0.1, &precision_);
    new ParamDouble(this, "smoothing", 33.33f, 0, 100, 0.1, &smoothing_);
    new ParamDouble(this, "q_", 8, 1, 16, 1, &q_);
    new ParamDouble(this, "threshold", 1e-4, 0, 1, 1e-4, &threshold_);
    new ParamInt   (this, "benchmark_n", 500, 1, 10000, 100, &benchmark_n );
}


void FFTGkfTest::process() {
    gpu_cache_clear();
    gpu_image src = gpuInput0();
    qDebug() << "src" << src.w() << src.h();

    publish("src", src);
    double sigma_r = (radius_-1) / precision_;
    fftgkf_t gkf(src.w(), src.h(), smoothing_/100.0f * sigma_r, sigma_r);
    gpu_timer tt;
    gpu_image dst = gkf(src, q_, threshold_);
    double t = tt.elapsed_time();
    qDebug() << "FFTGkfTest " << t << "ms";
    publish("dst", dst);
}


void FFTGkfTest::benchmark() {
    std::string name = QInputDialog::getText(NULL, "Benchmark", "ID:").toStdString();
    gpu_image src = gpuInput0();
    progress_t progress(0, 1, 0, 2);
    {
        progress_t progress(0, 1, 2, radius_+1);

        cpu_image T(radius_, benchmark_n, FMT_FLOAT);
        T.clear();
        for (int r = 2; r <= radius_; ++r) {
            if (!progress(r)) return;

            double sigma_r = (r-1) / precision_;
            gpu_cache_clear();
            fftgkf_t gkf(src.w(), src.h(), smoothing_/100.0f * sigma_r, sigma_r);

            progress_t progress2(r, r+1, 0, benchmark_n-1);
            double t0 = 0;
            gpu_timer tt;
            for (int k = 0; k < benchmark_n; k++) {
                if (!progress2(k)) return;
                tt.reset();
                gpu_image dst = gkf(src, q_, threshold_);
                double t = tt.elapsed_time();
                T.at<float>(r,k) = t;
                t0 += t;
            }
            qDebug() << "FFTGkfTest radius=" << r << t0 / benchmark_n;
        }
        csv_write(T, "fftgkf-" + name + "-fft.csv");
    }
    {
        progress_t progress(1, 2, 2, radius_+1);
        cpu_image T(radius_, benchmark_n, FMT_FLOAT);
        T.clear();
        for (int r = 2; r <= radius_; ++r) {
            if (!progress(r)) return;

            progress_t progress2(r, r+1, 0, benchmark_n-1);
            gpu_cache_clear();
            gpu_image krnl = circshift(gkf_create_kernel8x2(32, smoothing_ / 100.0f, precision_), 16, 16);
            gpu_timer tt;
            for (int k = 0; k < benchmark_n; k++) {
                if (!progress2(k)) return;
                tt.reset();
                gpu_image dst = gkf_opt8_filter2(src, krnl, r, q_, threshold_);
                double t = tt.elapsed_time();
                T.at<float>(r,k) = t;
            }
            qDebug() << "GkfTest radius=" << r;
        }
        csv_write(T, "fftgkf-" + name + "-tex.csv");
    }
}
