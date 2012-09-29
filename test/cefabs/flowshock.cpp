//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
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
#include "flowshock.h"
#include <oz/noise.h>
#include <oz/color.h>
#include <oz/bilateral.h>
#include <oz/gauss.h>
#include <oz/stgauss.h>
#include <oz/stgauss2.h>
#include <oz/fgauss.h>
#include <oz/st.h>
#include <oz/dog.h>
#include <oz/ssia.h>
//#include <oz/shock.h>
//#include <oz/shock2.h>
#include <oz/gkf.h>
#include <oz/gkf_kernel.h>
#include <oz/polygkf.h>
#include <oz/akf_opt.h>
#include <oz/shuffle.h>
#include <oz/beltrami.h>
#include <oz/blend.h>
#include <oz/resample.h>
#include <oz/hist.h>
#include <oz/ds.h>
//#include <oz/rst.h>
#include <oz/etf.h>
#include <oz/gpu_cache.h>
#include <oz/ivacef.h>
#include <cfloat>
using namespace oz;


extern gpu_image gpu_color_test( size_t w, size_t h );

//extern gpu_image gpu_st_relax_down( const gpu_image& src );
//extern gpu_image gpu_st_relax_up( const gpu_image& src0, const gpu_image& src1 );
//extern gpu_image gpu_st_jacobi_step(const gpu_image& src);
extern gpu_image gpu_st_relax( const gpu_image& src );

extern gpu_image gpu_st_adaptive_update( const gpu_image& R, const gpu_image& st,
                                         float sigma_min, float sigma_max );
extern gpu_image gpu_anisotropic_smooth( const gpu_image& src, const gpu_image& R );

gpu_image gpu_smooth_3( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2 );
gpu_image gpu_smooth_5( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2,
                        const gpu_image& src3, const gpu_image& src4 );
gpu_image gpu_smooth_5half( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2 );
gpu_image gpu_smooth_5half( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2 );
gpu_image gpu_colorize_sign( const gpu_image& src);
gpu_image gpu_warp_by_flow( const gpu_image& src, const gpu_image& flowU, const gpu_image& flowV );
gpu_image gpu_smooth_10half( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2,
                             const gpu_image& src3, const gpu_image& src4 );


gpu_image gpu_nagmat_sst( const gpu_image& st, const gpu_image& krnl, float radius, float q );

gpu_image gradient_bf( const gpu_image& src, const gpu_image& st,
                       float sigma_d, float sigma_r, bool adaptive );


void assertNaN(const gpu_image& img) {
    /*cpu_image<float4> c = img.cpu();
    for (int j = 0; j < (int)c.h(); ++j) {
        for (int i = 0; i < (int)c.w(); ++i) {
            float4 px = c(i,j);
            assert(!_isnan(px.x));
            assert(!_isnan(px.y));
            assert(!_isnan(px.z));
            assert(!_isnan(px.w));
        }
    }*/
}


FlowShock::FlowShock() {
    st_cache.setMaxCost(10);
    ParamGroup *g;

    g = new ParamGroup(this, "resample", false, &resample);
    new ParamInt(g, "resample_w", 640, 1, 4096, 1, &resample_w);
    new ParamInt(g, "resample_h", 480, 1, 4096, 1, &resample_h);

    g = new ParamGroup(this, "auto_levels", false, &auto_levels);
    new ParamDouble(g, "threshold", 0.1, 0, 100, 0.05, &auto_levels_threshold);

    new ParamInt(this, "total_N", 5, 1, 100, 1, &total_N);

    g = new ParamGroup(this, "noise removal", false, &remove_noise);
    new ParamChoice(g, "noise_technique", "bf", "bf|akf", &noise_technique);
    new ParamDouble(g, "noise_sigma_d", 2.0, 0.0, 10.0, 0.25, &noise_sigma_d);
    new ParamDouble(g, "noise_sigma_r", 4.25/100, 0.0, 100.0, 0.01, &noise_sigma_r);
    new ParamInt   (g, "noise_radius", 3, 1, 10, 1, &noise_radius);

    g = new ParamGroup(this, "structure tensor");
    new ParamChoice(g, "st_type", "sobel-rotopt3", "sobel-rotopt3|sobel-rotopt2|sobel-rotopt1|sobel|rst|ds-axis|etf-full|etf-xy|multi-scale", &st_type);
    new ParamChoice(g, "st_smooting", "isotropic", "isotropic|3x3|kf4|kuw|kf4-adaptive|adaptive|iso-red|nagmat|5x5|logxy|logbf|logp4|logp8|loggkf|p4|3x3X|logxy|bf|hourglass|gkf|akf|beltrami|none", &st_smoothing);
    new ParamDouble(g, "st_sigma_d", 1.0, 0.0, 10.0, 0.05, &st_sigma_d);
    new ParamDouble(g, "st_sigma_r", 0.8, 0.0, 100.0, 0.25, &st_sigma_r);
    new ParamInt   (g, "st_adaptive_N", 10, 0, 10000, 1, &st_adaptive_N);
    new ParamDouble(g, "st_adaptive_threshold", 0.04, 0.0, 1.0, 0.01, &st_adaptive_threshold);
    new ParamDouble(g, "st_q", 4.0, 0, 20, 0.25, &st_q);
    new ParamDouble(g, "st_relax", 0.002, 0.0, 1.0, 0.001, &st_relax);
    new ParamBool  (g, "st_flatten", false, &st_flatten);
    new ParamBool  (g, "st_normalize", false, &st_normalize);
    new ParamBool  (g, "ds_squared", true, &ds_squared);
    new ParamInt   (g, "st_mode", 0, 0, 1, 1, &st_mode);
    new ParamInt   (g, "moa_mode", 0, 0, 8, 1, &moa_mode);
    new ParamInt   (g, "etf_N", 3, 0, 10, 1, &etf_N);

    g = new ParamGroup(this, "flow smoothing", true, &fgauss);
    new ParamChoice(g, "fgauss_type", "rk2", "euler|rk2-nn|rk2|rk4", &fgauss_type);
    new ParamBool  (g, "fgauss_recalc_st", true, &fgauss_recalc_st);
    new ParamInt   (g, "fgauss_N", 1, 0, 100, 1, &fgauss_N);
    new ParamDouble(g, "fgauss_sigma_g", 0.0, 0.0, 20.0, 1, &fgauss_sigma_g);
    new ParamDouble(g, "fgauss_sigma_t", 6.0, 0.0, 20.0, 1, &fgauss_sigma_t);
    new ParamBool  (g, "fgauss_adaptive", true, &fgauss_adaptive);
    new ParamDouble(g, "fgauss_max", 22.5, 0.0, 90.0, 1, &fgauss_max);
    new ParamDouble(g, "fgauss_step", 1, 0.01, 2,  0.25, &fgauss_step);
    new ParamChoice(g, "bf_type", "none", "none|gradient|gradient-adaptive", &bf_type);
    new ParamDouble(g, "bf_sigma_d", 2.0, 0.0, 10.0, 0.05, &bf_sigma_d);
    new ParamDouble(g, "bf_sigma_r", 4.25, 0.0, 200.0, 0.25, &bf_sigma_r);

    g = new ParamGroup(this, "shock filter", true, &shock);
    new ParamChoice(g, "shock_type", "gradient", "gradient|weickert|fast-minmax|fast-minmax-xy|osher-sethian|kramer-bruckner|gradient-kramer|gradient-osher", &shock_type);
    new ParamBool  (g, "shock_recalc_st", true, &shock_recalc_st);
    new ParamDouble(g, "blur_sigma_i", 0.0, 0.0, 10.0, 0.25, &blur_sigma_i);
    new ParamDouble(g, "blur_sigma_g", 1.5, 0.0, 10.0, 0.25, &blur_sigma_g);
    new ParamDouble(g, "shock_radius", 2, 0.0, 10.0, 0.25, &shock_radius);
    new ParamChoice(g, "dog_type", "LoG", "LoG|LoG-color|FDoG|DoG", &dog_type );
    new ParamDouble(g, "dog_tau0", 1, -2, 2, 0.1, &dog_tau0);
    new ParamDouble(g, "dog_tau1", 0.005, -2, 2, 0.01, &dog_tau1);
    new ParamDouble(g, "dog_sigma_m", 0.0, 0.0, 10.0, 0.25, &dog_sigma_m);
    new ParamDouble(g, "dog_phi", 2.0, 0.0, 20.0, 0.5, &dog_phi);
    new ParamInt   (g, "shock_N", 3, 0, 100, 1, &weickert_N);
    new ParamDouble(g, "weickert_step", 0.4, 0, 10, 0.1, &weickert_step);

    g = new ParamGroup(this, "final smooth", true, &smooth);
    new ParamChoice(g, "smooth_type", "fgauss_rk", "fgauss_rk|fgauss|3x3|none", &smooth_type);
    new ParamDouble(g, "smooth_sigma", 1.5, 0.0, 10.0, 0.25, &smooth_sigma);
    new ParamDouble(g, "blend", 1.0, 0.0, 1.0, 0.1, &blend);

    g = new ParamGroup(this, "debug", false, &debug);
    new ParamBool(g, "draw_tf_fgauss", false, &draw_tf_fgauss);
    new ParamBool(g, "draw_tf_fgauss_relax", false, &draw_tf_fgauss_relax);
    new ParamBool(g, "draw_tfx_fgauss", false, &draw_tfx_fgauss);
    new ParamBool(g, "draw_tfA_fgauss", false, &draw_tfA_fgauss);
    new ParamBool(g, "draw_tf_shock", false, &draw_tf_shock);
    new ParamBool(g, "draw_fgauss", false, &draw_fgauss);
    new ParamBool(g, "draw_shock", false, &draw_shock);
    new ParamBool(g, "draw_st", false, &draw_st);

    OZ_CUDA_ERROR_CHECK();
    krnl81 = gkf_create_kernel1(32, 0.33f, 2.5f, 8);
    OZ_CUDA_ERROR_CHECK();
    krnl84 = gkf_create_kernel4(32, 0.33f, 2.5f, 8);
    OZ_CUDA_ERROR_CHECK();
}


QString FlowShock::caption() const {
    QString t;
    if (shock_type == "weickert")
        t += QString("Proposed algorithm with coherence-enhancing shock filter\n");
    else {
        if (fgauss_adaptive)
            t += QString("Proposed method\n");
        else
            t += QString("Proposed method (non-adaptive smoothing)\n");
    }

    t += QString("N=%1, ").arg(total_N);
    t += QString("tau_r=%1, ").arg(st_relax);
    t += QString("sigma_d=%1, ").arg(st_sigma_d);
    t += QString("sigma_t=%1, ").arg(fgauss_sigma_t);
    t += QString("sigma_i=%1, ").arg(blur_sigma_i);

    if (shock_type == "gradient"){
        t += QString("sigma_g=%1, ").arg(blur_sigma_g);
        t += QString("tau_s=%1").arg(dog_tau1);
    }
    if (shock_type == "weickert")
        t += QString("shock=weickert, shock_N=%1, shock_step=%2 ").arg(weickert_N).arg(weickert_step);

    /*if (!isDirty()) {
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, m_start, m_stop);
        if (cudaGetLastError() == cudaSuccess) {
            t += "\n";
            t += QString("%1 ms / %2 fps").arg(elapsedTime, 0, 'g', 5).arg(1000.0 / elapsedTime, 0, 'g', 4);
        }
    }*/

    return t;
}


void FlowShock::process() {
    gpu_image img = gpuInput0();
    if (resample) {
        int h = img.h() * resample_w / img.w();
        img = oz::resample(img, resample_w, h, RESAMPLE_LANCZOS2);
        gpu_cache_clear();
    }
    if (auto_levels) {
        img = oz::hist_auto_levels(img, auto_levels_threshold);
    }

    if (debug && (!noise.is_valid() || (noise.size() != img.size()))) {
        noise = noise_random(img.w(), img.h());
    }
    if (debug) publish("$source", img);
    if (debug) publish("gray0", rgb2gray(img));

    //timerStart();

    if (remove_noise) {
        if (noise_technique == "akf") {
            gpu_image st = st_scharr_3x3(img, 2.0);
            img = akf_opt_filter( img, st, krnl84, noise_radius, 8, 1, 1e-4f, 8);
        }
        if (noise_technique == "bf") {
            img = bilateral_filter(img, noise_sigma_d, noise_sigma_r );
        }
        if (debug) publish("img", img);
    }

    gpu_image st;
    gpu_image lfm;
    last_st = gpu_image();

    for (int k = 0; k < total_N; ++k) {
        bool lastN = (k == total_N - 1);
        if (fgauss) {
            if ((k == 0) || fgauss_recalc_st) {
                st = smoothedStructureTensor(img, 0, k);
                lfm = gpu_image();
            }
            /*if (debug && lastN) {
                if (!tfm.is_valid()) tfm = gpu_st_tfm(st);
                publish("fgauss-st", st);
                publish("fgauss-tfm", tfm);
                publish("fgauss-flow", gpu_st_colorize_flow(st, noise, 6));
                //publish("fgauss-flow4", gpu_st_colorize_flow(st, gpu_resize_bilinear(noise, 4*noise.w(), 4*noise.h()), 16));
                publish("fgauss-phi", gpu_st_colorize_angle(st));
                publish("fgauss-A", gpu_st_colorize_anisotropy(st));
                publish("fgauss-L1", gpu_shuffle(tfm, 2));
                publish("fgauss-L2", gpu_shuffle(tfm, 3));
                publish("fgauss-mag", gpu_st_colorize_mag(gpu_mul(st, 0.25f)));
            }*/

            /*if (fgauss_sigma_g > 0) {
                if (!tfm.is_valid()) tfm = gpu_st_tfm(st);
                img = gpu_directed_gauss(img, tfm, fgauss_sigma_g, true);
            }*/

            /*if ((bf_type == "gradient") || (bf_type == "gradient-adaptive")) {
                img = gradient_bf(img, st, bf_sigma_d, bf_sigma_r / 100.0f, (bf_type == "gradient-adaptive"));
                if (debug && lastN) publish("fgauss-bf", img);
            }*/

            for (int i = 0; i < fgauss_N; ++i) {
                if (fgauss_type == "euler") {
                    img = stgauss2_filter(img, st, fgauss_sigma_t, fgauss_max, fgauss_adaptive, true, true, 1, fgauss_step);
                } else if (fgauss_type == "rk2-nn") {
                    img = stgauss2_filter(img, st, fgauss_sigma_t, fgauss_max, fgauss_adaptive, false, false, 2, fgauss_step);
                } else if (fgauss_type == "rk2") {
                    img = stgauss2_filter(img, st, fgauss_sigma_t, fgauss_max, fgauss_adaptive, true, true, 2, fgauss_step);
                } else if (fgauss_type == "rk4") {
                    img = stgauss2_filter(img, st, fgauss_sigma_t, fgauss_max, fgauss_adaptive, true, true, 4, fgauss_step);
                }
            }
        }

        /*
        if (shock) {
            if (shock_type == "weickert") {
                for (int k = 0; k < weickert_N; ++k) {
                    st = smoothedStructureTensor(img, 1, k);
                    tfm = gpu_st_tfm(st);
                    gpu_image<float4> blur = img;
                    blur = gpu_gauss_filter_xy(blur, blur_sigma_i);
                    gpu_deriv_2nd_order_t d = gpu_deriv_2nd_order(blur);
                    gpu_image<float> sign = gpu_2nd_deriv_sign(tfm, d.Ixx, d.Ixy, d.Iyy);
                    img = gpu_shock_upwind(img, sign, weickert_step);
                }
            } else if (shock_type == "osher-sethian") {
                for (int k = 0; k < weickert_N; ++k) {
                    gpu_image<float4> img2 = img;
                    img2 = gpu_gauss_filter_xy(img2, blur_sigma_i);
                    img = gpu_osher_sethian_shock(img2, shock_radius);
                }
            } else if (shock_type == "kramer-bruckner") {
                for (int k = 0; k < weickert_N; ++k) {
                    gpu_image<float4> img2 = img;
                    img2 = gpu_gauss_filter_xy(img2, blur_sigma_i);
                    img = gpu_kramer_bruckner_shock(img2, shock_radius);
                }
            } else if (shock_type == "gradient-kramer") {
                for (int k = 0; k < weickert_N; ++k) {
                    st = smoothedStructureTensor(img, 1, k);
                    tfm = gpu_st_tfm(st);
                    gpu_image<float4> img2 = img;
                    img2 =  gpu_gauss_filter_xy(img2, blur_sigma_i);
                    img2 = gpu_directed_gauss( img2, tfm, blur_sigma_g, true);
                    img = gpu_gradient_kramer_bruckner(img2, tfm, shock_radius);
                }
            } else if (shock_type == "gradient-osher") {
                for (int k = 0; k < weickert_N; ++k) {
                    st = smoothedStructureTensor(img, 1, k);
                    tfm = gpu_st_tfm(st);
                    gpu_image<float4> img2 = img;
                    img2 =  gpu_gauss_filter_xy(img2, blur_sigma_i);
                    img2 = gpu_directed_gauss( img2, tfm, blur_sigma_g, true);
                    img = gpu_gradient_osher_sethian(img2, tfm, shock_radius);
                }
            } else {
                if (shock_recalc_st) {
                    st = smoothedStructureTensor(img, 1, k);
                    tfm = gpu_image<float4>();
                }

                if (!tfm.is_valid()) tfm = gpu_st_tfm(st);
                if (debug && lastN) {
                    publish("shock-tfm", tfm);
                    publish("shock-flow", gpu_st_colorize_flow(st, noise, 6));
                    publish("shock-phi", gpu_st_colorize_angle(st));
                    publish("shock-A", gpu_st_colorize_anisotropy(st));
                }

                gpu_image<float> dog;
                if (dog_type == "LoG-color") {
                    gpu_image<float4> L = img;
                    L =  gpu_gauss_filter_xy(L, blur_sigma_i);
                    dog = gpu_gradient_log4(L, tfm, blur_sigma_g);
                } else {
                    gpu_image<float> L = gpu_rgb2gray(img);
                    if (dog_type == "LoG") {
                        if (debug && lastN) publish("shock-L", L);
                        L =  gpu_gauss_filter_xy(L, blur_sigma_i);
                        dog = gpu_gradient_log(L, tfm, blur_sigma_g, dog_tau1);
                    } else if (dog_type == "FDoG") {
                        float s1 = blur_sigma_i;
                        float d = s1 - 1;
                        float s2 = sqrtf(2.56 + d*d);
                        dog = gpu_gradient_dog(L, tfm, s1, s2, dog_tau0, dog_tau1);
                        if (dog_sigma_m > 0) {
                            dog = gpu_fgauss_filter(dog, tfm, dog_sigma_m);
                        }
                    } else {
                        float s1 = blur_sigma_i;
                        float d = s1 - 1;
                        float s2 = sqrtf(2.56 + d*d);
                        dog = gpu_dog_filter(L, s1, s2, dog_tau0, dog_tau1);
                    }
                }

                if (debug && lastN) publish("dog", gpu_dog_colorize(dog));

                if (shock_type == "gradient") {
                    img = gpu_gradient_minmax_shock(img, tfm, dog, shock_radius);
                }
                else if (shock_type == "fast-minmax") {
                    img = gpu_fast_minmax_shock(img, dog, shock_radius);
                }
                else if (shock_type == "fast-minmax-xy") {
                    img = gpu_ssia_shock(img, dog);
                }
            }
        }
        */

        //if (dog.is_valid()) {
        //    dog = gpu_dog_threshold_tanh(dog, dog_phi);
        //    if (debug) publish("dog", dog);
        //}
    }

    if (smooth) {
        if (smooth_type == "3x3") {
            img = gauss_filter_3x3(img);
        }
        else if ((smooth_type == "fgauss")) {
            img = fgauss_filter(img, st_to_tangent(st), smooth_sigma);
        }
        else if ((smooth_type == "fgauss_rk")) {
            img = stgauss_filter(img, st, smooth_sigma, 90, false);
        }
    }

    //timerStop();
    publish("$result", img);

    /*
    QImage I(img.w(), 2*img.h(), QImage::Format_RGB32);
    QPainter p(&I);
    p.drawImage(0,0, input0());
    p.drawImage(0,img.h(), gpu_image_to_qimage(img));
    publish("both", I);
    */

    /*if (dog.is_valid()) {
        gpu_image<float4> white = gpu_util_set(make_float4(1,1,1,1), img.w(), img.h());
        publish("$result2", gpu_blend_intensity(gpu_lerp(white, img, blend), dog, GPU_BLEND_MULTIPLY));
    }*/
}


gpu_image FlowShock::smoothedStructureTensor(const gpu_image& img, /*const gpu_image<float>& dog,*/ int n, int k) {
#if 0
    if (st_type =="multi-scale") {
        gpu_image<float4> st = gpu_st_multi_scale( img, 5, st_sigma_d, GPU_RESAMPLE_GAUSSIAN, 0, st_mode, moa_mode );
        return st;
    }

    if (st_type =="ds-axis") {
        gpu_image<float> L = gpu_rgb2gray(img);
        gpu_image<float4> st = gpu_ds_scharr( L, st_sigma_d, st_normalize, ds_squared );
        return st;
    }

    if (st_type =="rst") {
        gpu_image<float> L = gpu_rgb2gray(img);
        return gpu_rst_scharr(L, st_sigma_d, st_sigma_r);
    }

    if (st_type =="etf-full") {
        gpu_image<float> L = gpu_rgb2gray(img);
        return gpu_st_from_tangent(gpu_etf_full(L, st_sigma_d, etf_N, 2));
    }

    if (st_type =="etf-xy") {
        gpu_image<float> L = gpu_rgb2gray(img);
        return gpu_st_from_tangent(gpu_etf_xy(L, st_sigma_d, etf_N, 2));
    }
#endif

    gpu_image st;
    /*if (st_type == "sobel-rotopt1") {
        st = gpu_st_sobel_rotopt(img);
        if (st_relax > 0) {
            st = gpu_st_threshold_mag(st, st_relax);
            st = gpu_st_relax(st);
        }
    } else if (st_type == "sobel-rotopt2") {
        st = gpu_st_sobel_rotopt2(img);
        if (st_relax > 0) {
            st = gpu_st_threshold_mag(st, st_relax);
            //st = gpu_st_relax(st);
        }
    } else if (st_type == "sobel-rotopt3") {
        st = gpu_st_sobel_rotopt3(img, last_st, st_relax);
        if (!last_st.is_valid()) {
            st = gpu_st_relax(st);
            if (st_flatten) {
                st = gpu_st_flatten(st);
            }
        }
    } else*/ {
        st = st_scharr_3x3(img);
        if (st_relax > 0) {
            //st = gpu_st_threshold_mag(st, st_relax);
            //st = gpu_st_relax(st);
        }
    }

#if 0
    if (st_normalize) {
        st = gpu_st_normalize(st);
    }

    if (st_smoothing == "p4") {
        st = gpu_polynomial_gkf_opt4(st, 0.5*st_sigma_d, st_q, 1.0/st_sigma_d, 0.84f);
    }
    else if (st_smoothing == "logp4") {
        assertNaN(st);
        st = gpu_st_log(st);
        assertNaN(st);
        st = gpu_polynomial_gkf_opt4(st, 0.5*st_sigma_d, st_q, 1.0/st_sigma_d, 0.84f);
        assertNaN(st);
        st = gpu_st_exp(st);
        assertNaN(st);
    }
    else if (st_smoothing == "logp8") {
        st = gpu_st_log(st);
        st = gpu_polynomial_gkf_opt8(st, 0.5*st_sigma_d, st_q, 1.0/st_sigma_d, 3.77f);
        st = gpu_st_exp(st);
    }
    else if (st_smoothing == "bf") {
        gpu_image<float4> lab = gpu_rgb2lab(img);
        st = gpu_joint_bilateral_filter(st, lab, st_sigma_d, st_sigma_r );
    }
    else if (st_smoothing == "logbf") {
        st = gpu_st_log(st);
        st = gpu_bilateral_filter(st, st_sigma_d, st_sigma_r );
        st = gpu_st_exp(st);
    }
    else if (st_smoothing == "logxy") {
        assertNaN(st);
        st = gpu_st_log(st);
        assertNaN(st);
        st = gpu_gauss_filter_xy(st, st_sigma_d);
        assertNaN(st);
        st = gpu_st_exp(st);
        assertNaN(st);
    }
    else if (st_smoothing == "loggkf") {
        st = gpu_st_log(st);
        st = gpu_gkf_filter(st, krnl81, 0.5*st_sigma_d, st_q);
        st = gpu_st_exp(st);
    }
    else if (st_smoothing == "adaptive") {
        st = gpu_st_smooth_adaptive_3x3( st, st_adaptive_threshold, st_adaptive_N );
    }
    else if (st_smoothing == "3x3X") {
        st = gpu_st_smooth_adaptive_3x3N( st, st_adaptive_threshold, st_adaptive_N );
    }
    else if (st_smoothing == "hourglass") {
        gpu_image<float4> tfm = gpu_st_tfm(gpu_gauss_filter_3x3(st));
        st = gpu_hourglass_gauss( st, tfm, st_sigma_d, 0.4f);
    }
    else if (st_smoothing == "gkf") {
        st = gpu_gkf_filter(st, krnl81, 0.5*st_sigma_d, st_q);
    }
    else if (st_smoothing == "akf") {
        gpu_image<float4> tfm = gpu_st_tfm(gpu_st_scharr(img, st_sigma_d));
        st = gpu_kernel_based_akf_opt8(st, tfm, krnl84, 0.5*st_sigma_d, st_q, 1);
    }
    else if (st_smoothing == "kf4") {
        st = gpu_st_gkf_p4(st, st_sigma_d, st_q );
    }
    else if (st_smoothing == "kf4-adaptive") {
        st = gpu_st_gkf_p4_adaptive(st, st_sigma_d, st_q, st_adaptive_threshold, st_adaptive_N );
    }
    else if (st_smoothing == "kuw") {
        st = gpu_st_kuwahara(st, ceil(0.5*st_sigma_d), st_q );
    }
    else if (st_smoothing == "nagmat") {
        st = gpu_nagmat_sst(st, krnl81, 0.5*st_sigma_d, st_q);
    }
    else if (st_smoothing == "isotropic") {
        st = gpu_gauss_filter_xy(st, st_sigma_d);
    }
    else if (st_smoothing == "iso-red") {
        st = gpu_gauss_filter_xy(st, st_sigma_d*0.75);
    }
    else if (st_smoothing == "3x3") {
        st = gpu_gauss_filter_3x3(st);
    }
    else if (st_smoothing == "5x5") {
        st = gpu_gauss_filter_5x5(st);
    }
    else if (st_smoothing == "beltrami") {
        st = gpu_beltrami(st, 0.005f, 0.21f);
    }

    /*if (st_temp_smooth) {
        gpu_image<float4> st0 = stFromCache(-2, n, k);
        gpu_image<float4> st1 = stFromCache(-1, n, k);
        if (st0.is_valid() && st1.is_valid()) {
            st = gpu_smooth_5half(st0, st1, st);
        }
        st_cache.insert((frame() << 8) + 2*k+n, new gpu_image<float4>(st));
    }*/
#endif

    assertNaN(st);
    last_st = st;
    return st;
}


/*gpu_image<float4> FlowShock::stFromCache(int f, int n, int k) {
    gpu_image<float4> *tmp = 0;
    if (frame() + f >= 0) tmp = st_cache.object(((frame() + f) << 8) + 2*k+n);
    if (!tmp) {
        return gpu_image<float4>();
    }
    return *tmp;
}*/

#if 0

struct draw_tfm_lic_t {
    size_t w;
    size_t h;
    float twoSigma2;
    float halfWidth;
    float2 p0;
    float2 v0;
    float sum;
    float2 v;
    float2 p;
    cpu_image<float4> tfm;
    QPolygonF P;

    draw_tfm_lic_t( float2 _p0, float sigma, cpu_image<float4>& _tfm, bool adaptive ) {
        tfm = _tfm;
        w = tfm.w();
        h = tfm.h();
        p0 = _p0;
        v0 = make_float2(tfm(p0.x, p0.y));
        sum = 1;
        float A = tfm2A(tfm(p0.x, p0.y));
        float sigmaA = adaptive? sigma * 0.25f * (1 + A) * (1 + A) : sigma;
        twoSigma2 = 2 * sigmaA * sigmaA;
        halfWidth = 2 * sigmaA;

        smooth(-1);
        P.append(QPointF(p0.x, p0.y));
        smooth(1);
    }

    void smooth( int sign ) {
        v = v0 * sign;
        p = p0 + v;
        float r = 1;
        while ((r < halfWidth) && (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  {
            if (sign == -1)
                P.prepend(QPointF(p.x, p.y));
            else
                P.append(QPointF(p.x, p.y));

            float2 t = make_float2(tfm(p.x, p.y));
            float vt = dot(v, t);
            if (vt == 0) break;
            if (vt < 0) t = -t;

            v = t;
            p += t;
            r += 1;
        }
    }
};

/*
struct draw_st_lic_t {
    size_t w;
    size_t h;
    float twoSigma2;
    float halfWidth;
    float2 p0;
    float2 v0;
    float2 v;
    float2 p;
    float stop;
    cpu_image<float4> st;
    QPolygonF P;
    bool bilinear;

    draw_st_lic_t(float2 _p0, float sigma, cpu_image<float4>& _st, bool adaptive, float max_angle, bool bilinear) {
        st = _st;
        w = st.w();
        h = st.h();
        p0 = _p0;
        float4 g = st(p0.x, p0.y);
        v0 = st2tangent(g);
        stop = cos(max_angle * M_PI / 180.0f);
        this->bilinear = bilinear;

        float A = tfm2A(st2tfm(g));
        float sigmaA = adaptive? sigma * 0.25f * (1 + A) * (1 + A) : sigma;
        twoSigma2 = 2 * sigmaA * sigmaA;
        halfWidth = 2 * sigmaA;

        smooth_rungekutta(-1);
        P.append(QPointF(p0.x, p0.y));
        smooth_rungekutta(1);
    }

    void smooth_rungekutta( float sign ) {
        v = v0 * sign;

        float2 t = st2tangent(st.sample_linear(p0.x + 0.5f * v.x, p0.y + 0.5f * v.y));
        float vt = dot(v, t);
        if (vt < 0) t = -t;
        v = t;
        p = p0 + v;

        float r = 1;
        while ((r < halfWidth) && (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  {
            if (sign == -1)
                P.prepend(QPointF(p.x, p.y));
            else
                P.append(QPointF(p.x, p.y));

            t = st2tangent(bilinear? st.sample_linear(p.x, p.y) : st(p.x, p.y));
            vt = dot(v, t);
            if (fabs(vt) <= stop) break;
            if (vt < 0) t = -t;

            t = st2tangent(bilinear? st.sample_linear(p.x + 0.5f * t.x, p.y + 0.5f * t.y) : st(p.x + 0.5f * t.x, p.y + 0.5f * t.y));
            vt = dot(v, t);
            if (fabs(vt) <= stop) break;
            if (vt < 0) t = -t;

            v = t;
            p += t;
            r += 1;
        }
    }
};
*/


struct XPoly : QPolygonF {
    XPoly(float sigma) {
        radius_ = 2 * sigma;
    }

    float radius() const {
        return radius_;
    }

    void operator()(int sign, float u, float2 p) {
        if (sign < 0)
            prepend(QPointF(p.x, p.y));
        else
            append(QPointF(p.x, p.y));
    }

    float radius_;
};


void FlowShock::drawTFM(const cpu_image<float4>& tfm, ImageView *view, QPainter &p, const QRect& R) {
    if (!tfm.is_valid())
        return;

    for (int j = R.top(); j <= R.bottom(); ++j) {
        for (int i = R.left(); i <= R.right(); ++i) {
            float4 t = tfm(i, j);
            QPointF q(i+0.5, j+0.5);
            QPointF v(0.45 * t.x, 0.45 * t.y);
            p.drawLine(q-v, q+v);
        }
    }
}


void FlowShock::drawTFM2(const cpu_image<float4>& st, ImageView *view, QPainter &p, const QRect& R) {
    if (!st.is_valid())
        return;

    QPen pblue(Qt::blue, view->pt2px(0.25f), Qt::SolidLine, Qt::RoundCap);
    QPen pred(Qt::red, view->pt2px(0.25f), Qt::SolidLine, Qt::RoundCap);

    for (int j = R.top(); j <= R.bottom(); ++j) {
        for (int i = R.left(); i <= R.right(); ++i) {
            float4 g = st(i, j);
            float4 t = st2tfm(g);

            if (g.w == 0) {
                p.setPen(pred);
            } else {
                p.setPen(pblue);
            }
            QPointF q(i+0.5, j+0.5);
            QPointF v(0.45 * t.x, 0.45 * t.y);
            p.drawLine(q-v, q+v);
        }
    }
}


void FlowShock::drawTFMx(const cpu_image<float4>& st, ImageView *view, QPainter &p, const QRect& R) {
    if (!st.is_valid())
        return;
    bool nearest = fgauss_type == "euler";

    p.setPen(QPen(Qt::red, view->pt2px(0.25f), Qt::SolidLine, Qt::RoundCap));
    int N = 3;
    float d = 0.35f / (2*N+1);
    for (int j = R.top(); j <= R.bottom(); ++j) {
        for (int i = R.left(); i <= R.right(); ++i) {
            for (int k = -N; k <= N; ++k) {
                for (int l = -N; l <= N; ++l) {
                    float x = i + 0.5f + (float)l / (2*N+1);
                    float y = j + 0.5f + (float)k / (2*N+1);

                    float4 g = nearest? st(x, y) : st.sample_linear(x, y);
                    float4 t = st2tfm(g);

                    QPointF q(x, y);
                    QPointF v(d * t.x, d * t.y);
                    p.drawLine(q-v, q+v);
                }
            }
        }
    }
}


void FlowShock::drawTFMA(const cpu_image<float4>& tfm, ImageView *view, QPainter &p, const QRect& R, float pt) {
    if (!tfm.is_valid())
        return;

    QPen tpen(Qt::blue, pt, Qt::SolidLine, Qt::RoundCap);
    QPen Apen(Qt::red, pt, Qt::SolidLine, Qt::RoundCap);
    for (int j = R.top(); j <= R.bottom(); ++j) {
        for (int i = R.left(); i <= R.right(); ++i) {
            float4 t = tfm(i, j);
            QPointF q(i+0.5, j+0.5);
            {
                p.setPen(tpen);
                QPointF v(0.45 * t.x, 0.45 * t.y);
                p.drawLine(q-v, q+v);
            }
            {
                p.setPen(Apen);
                float A = tfm2A(t);
                QPointF v(0.45 * A * t.y, -0.45 * A * t.x);
                p.drawLine(q-v, q+v);
            }
        }
    }
}


void FlowShock::drawTFME(const cpu_image<float4>& st, ImageView *view, QPainter &p, const QRect& R, float pt) {
    if (!st.is_valid())
        return;

    p.setPen(QPen(Qt::red, pt, Qt::SolidLine, Qt::RoundCap));
    for (int j = R.top(); j <= R.bottom(); ++j) {
        for (int i = R.left(); i <= R.right(); ++i) {
            p.save();
            p.translate((i+0.5), (j+0.5));

            float4 g = st(i, j);
#if 0
            if (_isnan(g.x) || _isnan(g.y) || _isnan(g.x)) {
                p.setPen(QPen(Qt::cyan, pt, Qt::SolidLine, Qt::RoundCap));
                p.drawLine(QPointF(-0.2,-0.2), QPointF(0.2,0.2));
                p.drawLine(QPointF(0.2, -0.2), QPointF(-0.2, 0.2));
            } else {
                float angle = st2angle(g);
                float2 l = st2lambda(g);

                float l1 = 0.25*sqrtf(fmaxf(0, l.x));
                float l2 = 0.25*sqrtf(fmaxf(0, l.y));

                p.rotate(180.0 * angle / M_PI);
                if (l2 > 0) {
                    p.drawEllipse(QPointF(0,0), l1, l2);
                } else {
                    if (l1 > 0) {
                        p.drawLine(QPointF(-l1,0), QPointF(l1,0));
                    } else {
                        p.setPen(QPen(Qt::yellow, pt, Qt::SolidLine, Qt::RoundCap));
                        p.drawLine(QPointF(-0.2,0), QPointF(0.2,0));
                        p.drawLine(QPointF(0, -0.2), QPointF(0, 0.2));
                    }
                }
            }
#endif

            p.restore();
        }
    }
}


static void draw_point(QPainter& p, QPointF& pos, float r, const QBrush& b, bool outline) {
    QPainterPath path;
    path.addEllipse(pos, r, r);
    p.fillPath(path, b);
    p.drawPath(path);
}
#endif


void FlowShock::draw(ImageView *view, QPainter &p, int pass) {
    QRect aR = p.clipBoundingRect().toAlignedRect();
    Module::draw(view, p, pass);
#if 0
    if (!debug) return;
    {
        cpu_image<float4> st;
        cpu_image<float4> tfm;
        if (draw_tf_fgauss && (view->zoom() >= 2)) {
            if (!tfm.is_valid())  tfm = getImage<float4>("fgauss-tfm");
            p.setPen(QPen(Qt::blue, view->pt2px(0.25), Qt::SolidLine, Qt::RoundCap));
            drawTFM(tfm, view, p , aR);
        }
        if (draw_tf_fgauss_relax && (view->zoom() >= 2)) {
            if (!st.is_valid())  st = getImage<float4>("fgauss-st");
            p.setPen(QPen(Qt::blue, view->pt2px(0.25), Qt::SolidLine, Qt::RoundCap));
            drawTFM2(st, view, p , aR);
        }
        if (draw_tfx_fgauss && (view->zoom() >= 50)) {
            if (!st.is_valid()) st = getImage<float4>("fgauss-st");
            drawTFMx(st, view, p , aR);
        }
        if (draw_tfA_fgauss && (view->zoom() >= 6)) {
            if (!tfm.is_valid())  tfm = getImage<float4>("fgauss-tfm");
            drawTFMA(tfm, view, p , aR, view->pt2px(0.25));
        }
        if (draw_st && (view->zoom() >= 8)) {
            if (!st.is_valid()) st = getImage<float4>("fgauss-st");
            drawTFME(st, view, p , aR, view->pt2px(0.25));
        }

        if (draw_fgauss && (view->zoom() >= 1)) {
            if (!tfm.is_valid()) tfm = getImage<float4>("fgauss-tfm");
            if (!st.is_valid()) st = getImage<float4>("fgauss-st");
            if (tfm.is_valid() && st.is_valid()) {
                if (fgauss_type == "euler") {
                    draw_tfm_lic_t L(make_float2(pos.x(), pos.y()), fgauss_sigma_t, tfm, fgauss_adaptive);
                    p.setPen(QPen(Qt::black, view->pt2px(1), Qt::SolidLine, Qt::RoundCap));
                    p.drawPolyline(L.P);
                    p.setPen(QPen(Qt::black, view->pt2px(0.25), Qt::SolidLine, Qt::RoundCap));
                    for (int i = 0; i < L.P.count(); ++i) {
                        draw_point( p, L.P[i], view->pt2px(1.5), (i == L.P.count()/2)? Qt::darkRed : Qt::black , true );
                    }
                } else {
                    std::vector<float3> P3 = gpu_stgauss2_path(
                                        pos.x(), pos.y(),
                                        st, fgauss_sigma_t, fgauss_max, fgauss_adaptive,
                                        (fgauss_type == "rk2-nn")? false : true,
                                        (fgauss_type == "rk4")? 4 : 2, fgauss_step);

                    QPolygonF P;
                    for (int i = 0; i < (int)P3.size(); ++i) {
                        P.append(QPointF(P3[i].x, P3[i].y));
                    }

                    p.setPen(QPen(Qt::black, view->pt2px(1), Qt::SolidLine, Qt::RoundCap));
                    p.drawPolyline(P);
                    p.setPen(QPen(Qt::black, view->pt2px(0.25), Qt::SolidLine, Qt::RoundCap));
                    for (int i = 0; i < P.count(); ++i) {
                        draw_point( p, P[i], view->pt2px(1.5), (i == P.count()/2)? Qt::darkRed : Qt::black , true );
                    }
                }
            }
        }
    }
    {
        cpu_image<float4> tfm;
        cpu_image<float> dog;
        if (draw_tf_shock && (view->zoom() >= 8)) {
            if (!tfm.is_valid()) tfm = getImage<float4>("shock-tfm");
            p.setPen(QPen(Qt::blue, view->pt2px(0.25), Qt::SolidLine, Qt::RoundCap));
            drawTFM(tfm, view, p , aR);
        }

        if (draw_shock && (view->zoom() >= 8)) {
            if (!tfm.is_valid())  tfm = getImage<float4>("shock-tfm");
            if (!dog.is_valid())  dog = getImage<float>("shock-dog");

            QPolygonF P0;
            QPolygonF P1;
            P0.append(QPointF(pos.x(), pos.y()));
            P1.append(QPointF(pos.x(), pos.y()));

            float4 t = tfm(pos.x(), pos.y());
            float2 n = make_float2(t.y, -t.x);
            if (dot(n,n) > 0) {
                float2 nabs = fabs(n);
                float ds;
                float2 dp;
                if (nabs.x > nabs.y) {
                    ds = 1.0f / nabs.x;
                    dp = make_float2(0, 0.5f);
                } else {
                    ds = 1.0f / nabs.y;
                    dp = make_float2(0.5f, 0);
                }

                for (int k = -1; k <= 1; k += 2) {
                    for (float d = ds; d < shock_radius; d += ds) {
                        float2 o = k*d*n;
                        if (k == -1)
                            P0.prepend(QPointF(pos.x() + o.x, pos.y() + o.y));
                        else
                            P0.append(QPointF(pos.x() + o.x, pos.y() + o.y));

                        P1.append(QPointF(floor(pos.x() + o.x + dp.x)+0.5, floor(pos.y() + o.y + dp.y) + 0.5));
                        P1.append(QPointF(floor(pos.x() + o.x - dp.x)+0.5, floor(pos.y() + o.y - dp.y) + 0.5));
                    }
                }
            }

            p.setPen(QPen(Qt::yellow, view->pt2px(2), Qt::SolidLine, Qt::RoundCap));
            p.drawPolyline(P0);

            for (int j = aR.top(); j <= aR.bottom(); ++j) {
                for (int i = aR.left(); i <= aR.right(); ++i) {
                    int sign = dog(i, j) <= 0.5;
                    QPointF a = QPointF(i+0.5f, j+0.5f);
                    Qt::GlobalColor color = sign? Qt::darkGreen : Qt::darkRed;
                    p.setPen(QPen(color, view->pt2px(1.25f), Qt::SolidLine, Qt::FlatCap));
                    p.drawLine(a+QPointF(0.15f,0), a+QPointF(-0.15,0));
                    if (sign) p.drawLine(a+QPointF(0,0.15f), a+QPointF(0,-0.15f));
                }
            }

            for (int i = 0; i < P1.count(); ++i) {
                QPointF a = P1[i];

                int sign = dog(a.x(), a.y()) <= 0.5;
                Qt::GlobalColor color = sign? Qt::darkGreen : Qt::darkRed;
                p.setPen(QPen(color, view->pt2px(1.5f), Qt::SolidLine, Qt::FlatCap));
                p.drawEllipse(a, 0.3f, 0.3f);
                //p.setPen(QPen(color, view->pt2px(0.5f), Qt::SolidLine, Qt::FlatCap));
                //p.drawLine(a+QPointF(0.15f,0), a+QPointF(-0.15,0));
                //if (sign) p.drawLine(a+QPointF(0,0.15f), a+QPointF(0,-0.15f));
            }
        }
    }
#endif
}


void FlowShock::dragBegin(ImageView *view, QMouseEvent* e) {
    QPointF pos = view->view2image(e->pos());
    this->pos = QPointF( floor(pos.x())+.5f, floor(pos.y())+.5f );
    setDirty();
    check();
    //qDebug() << QString("clicked (%1, %2)").arg(pos.x()).arg(pos.y());
}

