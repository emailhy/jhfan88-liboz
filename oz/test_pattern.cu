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
#include <oz/test_pattern.h>
#include <oz/generate.h>
#include <oz/colormap_util.h>

namespace oz {

    struct TestCircle : public generator<float> {
        int width_;
        int height_;
        float r_;

        TestCircle( int width, int height, float r )
            : width_(width), height_(height), r_(r) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float x = ix - 0.5f * width_ + 0.5f;
            float y = iy - 0.5f * height_ + 0.5f;
            float r = sqrtf(x*x + y*y);
            return (r < r_)? 1 : 0;
        }
    };

    gpu_image test_circle( int width, int height, float r ) {
        return generate(width, height, TestCircle(width, height, r));
    }


    struct TestWiggle : public generator<float> {
        int width_;
        int height_;
        float r_;

        TestWiggle( int width, int height, float r )
            : width_(width), height_(height), r_(r) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float x = ix - 0.5f * width_ + 0.5f;
            float y = iy - 0.5f * height_ + 0.5f;
            float phi = atan2f(y, x);
            float r = sqrtf(x*x + y*y) + 10 * __sinf(10 * phi);
            return (r < r_)? 1 : 0;
        }
    };

    gpu_image test_wiggle( int width, int height, float r ) {
        return generate(width, height, TestWiggle(width, height, r));
    }


    struct TestLine : public generator<float> {
        int width_;
        int height_;
        float phi_;
        float r_;

        TestLine( int width, int height, float phi, float r )
            : width_(width), height_(height), phi_(phi), r_(r) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float x = ix - 0.5f * width_ + 0.5f;
            float y = iy - 0.5f * height_ + 0.5f;
            float d = y * cosf(phi_) - x * sinf(phi_);
            return (fabs(d) < r_)? 1 : 0;
        }
    };

    gpu_image test_line( int width, int height, float phi, float r ) {
        return generate(width, height, TestLine(width, height, radians(phi), r));
    }


    struct TestSimple : public generator<float> {
        int width_;
        int height_;
        float phi_;
        float phase_;
        float scale_;
        int function_;

        TestSimple( int width, int height, float phi, float phase, float scale, int function )
            : width_(width), height_(height), phi_(phi), phase_(phase), scale_(scale), function_(function) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float x = ix - 0.5f * width_ + 0.5f;
            float y = iy - 0.5f * height_ + 0.5f;
            float d = x * cosf(phi_) + y * sinf(phi_);
            float u = 2 * CUDART_PI_F * phase_ * d / width_;
            float f = 0;
            switch(function_) {
                case 0:
                    f = cosf(u);
                    break;
                case 1:
                    f = sinf(u);
                    break;
                case 2:
                    f = (u != 0)? sinf(u)/u : 1;
                    break;
            }
            return 0.5f + 0.5f * scale_ * f;
        }
    };

    gpu_image test_simple( int width, int height, float phi, float phase, float scale, int function ) {
        return generate(width, height, TestSimple(width, height, radians(phi), phase, scale, function));
    }


    struct TestSphere : public generator<float> {
        int width_;
        int height_;

        TestSphere( int width, int height )
            : width_(width), height_(height){}

        inline __device__ float operator()( int ix, int iy ) const {
            float x = 2 * (ix + 0.5f) / width_ - 1;
            float y = 2 * (iy + 0.5f) / height_ - 1;
            float r = sqrtf(x*x + y*y);
            return fmaxf(0, 1-r);
        }
    };

    gpu_image test_sphere( int width, int height ) {
        return generate(width, height, TestSphere(width, height));
    }


    struct TestGrad3 : public generator<float3> {
        int width_;
        int height_;

        TestGrad3( int width, int height )
            : width_(width), height_(height){}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float x = ix - 0.5f * width_ + 0.5f;
            float y = iy - 0.5f * height_ + 0.5f;
            float c[3];
            float phi = CUDART_PIO2_F;
            for (int k = 0; k < 3; ++k) {
                float r = (x * cosf(phi) + y * sinf(phi)) / width_;
                c[k] = 0.5f+r;
                phi += 2*CUDART_PI_F/3;
            }
            return make_float3(c[0], c[1], c[2]);
        }
    };

    gpu_image test_grad3( int width, int height ) {
        return generate(width, height, TestGrad3(width, height));
    }


    struct TestKnutssonRing : public generator<float2> {
        inline __device__ float2 operator()( int ix, int iy) const {
            float x = ix - 0.5f * 512 + 0.5f;
            float y = iy - 0.5f * 512 + 0.5f;

            float r = sqrtf( x*x + y*y );
            float m = ((r >= 56) && (r < 256))? 1 : 0;

            return make_float2(ftest(r) / 254.0f, m);
        }

        inline __device__ float g(float r) const {
            return sinf( 112*CUDART_PI_F / logf(2) * (powf(2,-r/56) - powf(2,-256.0f/56)));
        }

        inline __device__ float ftest(float r) const {
            if (r < 56) return 127;
            if (r < 64) return 127 * (1+g(r)*cos(CUDART_PI_F*r/16 - 4*CUDART_PI_F)*cos(CUDART_PI_F*r/16 - 4*CUDART_PI_F));
            if (r < 224) return 127 * (1+g(r));
            if (r < 255) return 127 * (1+g(r)*sin(CUDART_PI_F*r/64 - 4*CUDART_PI_F)*sin(CUDART_PI_F*r/64 - 4*CUDART_PI_F));
            return 127;
        }
    };

    gpu_image test_knutsson_ring() {
        return generate(512, 512, TestKnutssonRing());
    }


    struct TestJaenichRing : public generator<float2> {
        int width_;
        int height_;
        float g0_;
        float km_;
        float rm_;
        float w_;

        TestJaenichRing( int width, int height, float g0, float km, float rm, float w )
            : width_(width), height_(height), g0_(g0), km_(km), rm_(rm), w_(w) {}

        inline __device__ float2 operator()( int ix, int iy ) const {
            float x = ix - 0.5f * width_ + 0.5f;
            float y = iy - 0.5f * height_ + 0.5f;

            float t = sqrtf(x*x + y*y);
            float f = (0.5f * tanhf((rm_ - t) / w_) + 0.5f);
            float v = g0_ * f * sinf(0.5f * km_ * CUDART_PI_F * t * t / rm_);

            float m = (t < rm_ + w_)? 1 : 0;

            return make_float2(0.5f + 0.5f * v, m);
        }
    };

    gpu_image test_jaenich_ring( int width, int height, float g0, float km, float rm, float w ) {
        return generate(width, height, TestJaenichRing(width, height, g0, km, rm, w));
    }


    struct TestZonePlate : public generator<float2> {
        int width_;
        int height_;
        float g0_;
        float km_;
        float rm_;
        float w_;
        bool inverted_;

        TestZonePlate( int width, int height, float g0, float km, float rm, float w, bool inverted )
            : width_(width), height_(height), g0_(g0), km_(km), rm_(rm), w_(w), inverted_(inverted) {}

        inline __device__ float2 operator()( int ix, int iy ) const {
            float x = ix - 0.5f * width_ + 0.5f;
            float y = iy - 0.5f * height_ + 0.5f;

            float r = sqrtf(x*x + y*y);
            float z = 0;
            if (r < rm_ + 2*w_) {
                float f = (0.5f * tanhf((rm_ - r) / w_) + 0.5f);
                if (inverted_) r = rm_ - r;
                z = g0_ * f * sinf(CUDART_PI * km_ * r * r  / rm_ / 2.0f);
            }

            return make_float2(0.5f + 0.5f * z, (sqrtf(x*x + y*y) < 0.475f*width_)? 1 : 0);
        }
    };

    gpu_image test_zoneplate( int width, int height, float g0, float km, float rm, float w, bool inverted ) {
        return generate(width, height, TestZonePlate(width, height, g0, km, rm, w, inverted));
    }


    struct TestColorFan : public generator<float3> {
        int width_;
        int height_;

        TestColorFan( int width, int height )
            : width_(width), height_(height) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float x = ix - 0.5f * width_ + 0.5f;
            float y = iy - 0.5f * height_ + 0.5f;
            float t = 0.5f + 0.5f * atan2f(y,x) / CUDART_PI;
            return colormap_H(t);
        }
    };

    gpu_image test_color_fan( int width, int height ) {
        return generate(width, height, TestColorFan(width, height));
    }

}
