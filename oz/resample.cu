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
#include <oz/resample.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace oz{

    template <typename T, typename Filter, int pass> struct UpSampler;

    template <typename T, typename Filter> struct UpSampler<T,Filter,0> : public generator<T> {
        gpu_sampler<T,0> src_;
        float scale_;
        Filter f_;
        UpSampler( const gpu_image& src, float scale, Filter f ) : src_(src), scale_(scale), f_(f) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float center = (ix + 0.5f) / scale_ - 0.5f;
            int left = (int)floor(center - f_.width());
            int right = (int)ceil(center + f_.width());

            T c = make_zero<T>();
            float sum = 0;
            for (int j = left; j <= right; ++j) {
                float w = f_(j - center);
                c += w * src_(j, iy);
                sum += w;
            }
            return c / sum;
        }
    };

    template <typename T, typename Filter> struct UpSampler<T,Filter,1> : public generator<T> {
        gpu_sampler<T,0> src_;
        float scale_;
        Filter f_;
        UpSampler( const gpu_image& src, float scale, Filter f ) : src_(src), scale_(scale), f_(f) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float center = (iy + 0.5f) / scale_ - 0.5f;
            int left = (int)floor(center - f_.width());
            int right = (int)ceil(center + f_.width());

            T c = make_zero<T>();
            float sum = 0;
            for (int j = left; j <= right; ++j) {
                float w = f_(j - center);
                c += w * src_(ix, j);
                sum += w;
            }
            return c / sum;
        }
    };


    template <typename T, typename Filter, int pass> struct DownSampler;

    template <typename T, typename Filter> struct DownSampler<T,Filter,0> : public generator<T> {
        gpu_sampler<T,0> src_;
        float scale_;
        Filter f_;
        DownSampler( const gpu_image& src, float scale, Filter f ) : src_(src), scale_(scale), f_(f) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float fwidth = f_.width() / scale_;
            float center = (ix + 0.5f) / scale_ - 0.5f;
            int left = (int)floor(center - fwidth);
            int right = (int)ceil(center + fwidth);

            T c = make_zero<T>();
            float sum = 0;
            for (int j = left; j <= right; ++j) {
                float w = f_((j - center) * scale_);
                c += w * src_(j, iy);
                sum += w;
            }
            return c / sum;
        }
    };

    template <typename T, typename Filter> struct DownSampler<T,Filter,1> : public generator<T> {
        gpu_sampler<T,0> src_;
        float scale_;
        Filter f_;
        DownSampler( const gpu_image& src, float scale, Filter f ) : src_(src), scale_(scale), f_(f) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float fwidth = f_.width() / scale_;
            float center = (iy + 0.5f) / scale_ - 0.5f;
            int left = (int)floor(center - fwidth);
            int right = (int)ceil(center + fwidth);

            T c = make_zero<T>();
            float sum = 0;
            for (int j = left; j <= right; ++j) {
                float w = f_((j - center) * scale_);
                c += w * src_(ix, j);
                sum += w;
            }
            return c / sum;
        }
    };


    struct BoxFilter {
        __device__ float width() const { return 0.5f; }
        __device__ float operator()(float t) const {
            if ((t > -0.5f) && (t <= 0.5f)) return 1;
            return 0;
        }
    };


    struct TriangleFilter {
        __device__ float width() const { return 1; }
        __device__ float operator()(float t) const {
            if (t < 0) t = -t;
            if (t < 1) return 1 - t;
            return 0;
        }
    };


    struct QuadraticFilter {
        float R;
        QuadraticFilter(float R) { this->R = R; }
        __device__ float width() const { return 1.5f; }
        __device__ float operator()(float t) const {
            if (t < 0) t = -t;
            if (t < 1.5f) {
                float tt = t * t;
                if (t <= 0.5f)
                    return (-2 * R) * tt + 0.5f * (R + 1);
                else
                    return (R * tt) + (-2 * R - 0.5f) * t + (3.0f / 4.0f) * (R + 1);
            }
            return 0;
        }
    };


    struct BellFilter {
        __device__ float width() const { return 1.5f; }
        __device__ float operator()(float t) const {
            if (t < 0) t = -t;
            if (t < 0.5f) return 0.75f - t * t;
            if (t < 1.5f) {
                t = t - 1.5f;
                return 0.5f * t * t;
            }
            return 0;
        }
    };


    struct BSplineFilter {
        __device__ float width() const { return 2; }
        __device__ float operator()(float t) const {
            if (t < 0) t = -t;
            if (t < 1) {
                float tt = t * t;
                return (0.5f * tt * t) - tt + 2.0f / 3.0f;
            }
            if (t < 2) {
                t = 2 - t;
                return 1.0f / 6.0f * t * t * t;
            }
            return 0;
        }
    };


    // http://code.google.com/p/imageresampler/source/browse/trunk/resampler.cpp
    static inline __device__ float clean(float t){
       const float EPSILON = 0.0000125f;
       return (fabs(t) < EPSILON)? 0 : t;
    }


    static inline __device__ float sinc(float t) {
        t *= CUDART_PI_F;
        if ((t > -0.01f) && (t < 0.01f))
            return 1.0f + t*t*(-1.0f/6.0f + t*t*1.0f/120.0f);
        return __sinf(t) / t;
    }


    static inline __device__ float  blackman_window(float t) {
       return 0.42659071f + 0.49656062f * __cosf(CUDART_PI_F * t) + 0.07684867f * __cosf(2 * CUDART_PI_F * t);
    }


    struct Lanczos2Filter {
        __device__ float width() const { return 2; }
        __device__ float operator()(float t) const {
            if (t < 0) t = -t;
            if (t < 2) return clean(sinc(t) * sinc(t / 2));
            return 0;
        }
    };


    struct Lanczos3Filter {
        __device__ float width() const { return 3; }
        __device__ float operator()(float t) const {
            if (t < 0) t = -t;
            if (t < 3) return clean(sinc(t) * sinc(t / 3));
            return 0;
        }
    };


    struct BlackmanFilter {
        __device__ float width() const { return 3; }
        __device__ float operator()(float t) const {
            if (t < 0) t = -t;
            if (t < 3) return clean(sinc(t) * blackman_window(t / 3));
            return 0;
        }
    };


    struct MitchellFilter {
        float B, C;
        MitchellFilter(float B, float C) { this->B = B; this->C = C; }
        __device__ float width() const { return 2; }
        __device__ float operator()(float t) const {
            float tt = t * t;
            if (t < 0) t = -t;
            if (t < 1) {
                return ((12 - 9 * B - 6 * C) * t * tt +
                        (-18 + 12 * B + 6 * C) * tt +
                        (6 - 2 * B)
                       ) / 6;
            } else if ( t < 2) {
                return ((-1 * B - 6 * C) * t * tt +
                        (6 * B + 30 * C) * tt +
                        (-12 * B - 48 * C) * t +
                        (8 * B + 24 * C)
                       ) / 6;
            }
            return 0;
        }
    };


    struct GaussianFilter {
        __device__ float width() const { return 1.25f; }
        __device__ float operator()(float t) const {
            if (t < 0) t = -t;
            if (t < 1.25f)
                return __expf( -2 *t *t ) * blackman_window(t / 1.25f);
            return 0;
        }
    };


    struct GaussianFilter2 {
        float twoSigma2_;
        float width_;

        GaussianFilter2( float sigma, float precision ) {
            twoSigma2_ =  2 * sigma * sigma;
            width_ = sigma * precision;
        }

        __device__ float width() const { return width_; }

        __device__ float operator()(float t) const {
            return (twoSigma2_ > 1e-3)? __expf( -t*t / twoSigma2_ ) : 1;
        }
    };


    struct KaiserFilter {
        static __device__ float bessel0(double x) {
           const float EPSILON_RATIO = 1e-8;
           float xh, sum, pow, ds;
           int k;

           xh = 0.5f * x;
           sum = 1;
           pow = 1;
           k = 0;
           ds = 1;
           while (ds > sum * EPSILON_RATIO) {
              ++k;
              pow = pow * (xh / k);
              ds = pow * pow;
              sum = sum + ds;
           }

           return sum;
        }

        static __device__ double kaiser(double alpha, double half_width, double x) {
           const double ratio = (x / half_width);
           return bessel0(alpha * sqrtf(1 - ratio * ratio)) / bessel0(alpha);
        }

        __device__ float width() const { return 3; }
        __device__ float operator()(float t) const {
            const float KAISER_ALPHA = 4;
            if (t < 0)
            t = -t;

            if (t < 3) {
                // db atten
                const float att = 40.0f;
                const float alpha = (expf(logf(0.58417f * (att - 20.96f)) * 0.4f) + 0.07886f * (att - 20.96f));
                return clean(sinc(t) * kaiser(KAISER_ALPHA, 3, t));
            }

            return 0;
        }
    };


    template <typename T, typename Filter>
    static gpu_image resample_xy( const gpu_image& src, unsigned w, unsigned h, Filter f ) {
        float sx = (float)w / src.w();
        float sy = (float)h / src.h();

        gpu_image tmp;
        if (sx >= 1) {
            tmp = generate(w, src.h(), UpSampler<T,Filter,0>(src, sx, f));
        } else {
            tmp = generate(w, src.h(), DownSampler<T,Filter,0>(src, sx, f));
        }

        gpu_image dst;
        if (sy >= 1) {
            dst = generate(w, h, UpSampler<T,Filter,1>(tmp, sy, f));
        } else {
            dst = generate(w, h, DownSampler<T,Filter,1>(tmp, sy, f));
        }

        return dst;
    }


    template <typename T>
    static gpu_image  resampleT( const gpu_image& src, unsigned w, unsigned h, resample_mode_t mode ) {
        switch (mode) {
            case RESAMPLE_BOX:              return resample_xy<T>(src, w, h, BoxFilter());
            case RESAMPLE_TRIANGLE:         return resample_xy<T>(src, w, h, TriangleFilter());
            case RESAMPLE_BELL:             return resample_xy<T>(src, w, h, BellFilter());
            case RESAMPLE_QUADRATIC:        return resample_xy<T>(src, w, h, QuadraticFilter(1));
            case RESAMPLE_QUADRATIC_APPROX: return resample_xy<T>(src, w, h, QuadraticFilter(0.5f));
            case RESAMPLE_QUADRATIC_MIX:    return resample_xy<T>(src, w, h, QuadraticFilter(0.8f));
            case RESAMPLE_BSPLINE:          return resample_xy<T>(src, w, h, BSplineFilter());
            case RESAMPLE_LANCZOS2:         return resample_xy<T>(src, w, h, Lanczos2Filter());
            case RESAMPLE_LANCZOS3:         return resample_xy<T>(src, w, h, Lanczos3Filter());
            case RESAMPLE_BLACKMAN:         return resample_xy<T>(src, w, h, BlackmanFilter());
            case RESAMPLE_CUBIC:            return resample_xy<T>(src, w, h, MitchellFilter(1,0));
            case RESAMPLE_CATROM:           return resample_xy<T>(src, w, h, MitchellFilter(0,0.5f));
            case RESAMPLE_MITCHELL:         return resample_xy<T>(src, w, h, MitchellFilter(1.0f/3.0f, 1.0f/3.0f));
            case RESAMPLE_GAUSSIAN:         return resample_xy<T>(src, w, h, GaussianFilter());
            case RESAMPLE_KAISER:           return resample_xy<T>(src, w, h, KaiserFilter());
            default:
                OZ_X() << "Unsupported resampling mode";
        }
    }


    gpu_image resample( const gpu_image& src, unsigned w, unsigned h, resample_mode_t mode ) {
        switch (src.format()) {
            case FMT_FLOAT:  return resampleT<float >(src, w, h, mode);
            case FMT_FLOAT2: return resampleT<float2>(src, w, h, mode);
            case FMT_FLOAT3: return resampleT<float3>(src, w, h, mode);
            case FMT_FLOAT4: return resampleT<float4>(src, w, h, mode);
            default:
                OZ_INVALID_FORMAT();
        }
    }


    gpu_image resample_gaussian( const gpu_image& src, unsigned w, unsigned h, float sigma, float precision ) {
        switch (src.format()) {
            case FMT_FLOAT:  return resample_xy<float >(src, w, h, GaussianFilter2(sigma, precision));
            case FMT_FLOAT2: return resample_xy<float2>(src, w, h, GaussianFilter2(sigma, precision));
            case FMT_FLOAT3: return resample_xy<float3>(src, w, h, GaussianFilter2(sigma, precision));
            case FMT_FLOAT4: return resample_xy<float4>(src, w, h, GaussianFilter2(sigma, precision));
            default:
                OZ_INVALID_FORMAT();
        }
    }


    gpu_image resample_boxed( const gpu_image& src, unsigned max_w, unsigned max_h ) {
        gpu_image img = src;
        if (((int)img.w() > max_w) || ((int)img.h() > max_h)) {
            double zw = 1.0 * std::min<int>((int)img.w(), max_w) / img.w();
            double zh = 1.0 * std::min<int>((int)img.h(), max_h) / img.h();
            int w, h;
            if (zw <= zh) {
                w = max_w;
                h = (int)(zw * img.h());
            } else {
                w = (int)(zh * img.w());
                h = max_h;
            }
            img = resample(img, w, h, RESAMPLE_LANCZOS3);
        }
        return img;
    }
}
