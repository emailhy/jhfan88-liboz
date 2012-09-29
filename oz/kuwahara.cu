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
#include <oz/kuwahara.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {
    template<typename T> static __device__ float compoment_sum(T);
    template<> static inline __device__ float compoment_sum(float v) { return v; }
    template<> static inline __device__ float compoment_sum(float3 v) { return v.x + v.y + v.z; }

    template<typename T> struct KuwaharaFilter : public oz::generator<T> {
        gpu_sampler<T,0> src_;
        int radius_;

        KuwaharaFilter( const oz::gpu_image& src, int radius ) : src_(src), radius_(radius) {}

        inline __device__ T operator()( int ix, int iy ) const {
            int n = (radius_ + 1) * (radius_ + 1);

            T m[4];
            T s[4];
            for (int k = 0; k < 4; ++k) {
                m[k] = make_zero<T>();
                s[k] = make_zero<T>();
            }

            for (int j = -radius_; j <= 0; ++j)  {
                for (int i = -radius_; i <= 0; ++i)  {
                    T c = src_(ix + i, iy + j);
                    m[0] += c;
                    s[0] += c * c;
                }
            }

            for (int j = -radius_; j <= 0; ++j)  {
                for (int i = 0; i <= radius_; ++i)  {
                    T c = src_(ix + i, iy + j);
                    m[1] += c;
                    s[1] += c * c;
                }
            }

            for (int j = 0; j <= radius_; ++j)  {
                for (int i = 0; i <= radius_; ++i)  {
                    T c = src_(ix + i, iy + j);
                    m[2] += c;
                    s[2] += c * c;
                }
            }

            for (int j = 0; j <= radius_; ++j)  {
                for (int i = -radius_; i <= 0; ++i)  {
                    T c = src_(ix + i, iy + j);
                    m[3] += c;
                    s[3] += c * c;
                }
            }

            float min_sigma2 = 1e+2;
            T result;
            for (int k = 0; k < 4; ++k) {
                m[k] /= n;
                s[k] = fabs(s[k] / n - m[k] * m[k]);

                float sigma2 = compoment_sum(s[k]);
                if (sigma2 < min_sigma2) {
                    min_sigma2 = sigma2;
                    result = m[k];
                }
            }
            return result;
        }
    };
}


oz::gpu_image oz::kuwahara_filter( const gpu_image& src, int radius, int N ) {
    if (radius <= 0) return src;
    gpu_image img = src;
    for (int k = 0; k < N; ++k) {
        switch (img.format()) {
            case FMT_FLOAT:
                img = generate(img.size(), KuwaharaFilter<float >(img, radius));
                break;
            case FMT_FLOAT3:
                img = generate(img.size(), KuwaharaFilter<float3>(img, radius));
                break;
            default:
                OZ_INVALID_FORMAT();
        }
    }
    return img;
}
