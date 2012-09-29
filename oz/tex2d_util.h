//
// Based on NVIDIA GPU Computing SDK example.
// See GPU Gems 2: "Fast Third-Order Texture Filtering", Sigg & Hadwiger
// http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter20.html
// Reformulation thanks to Keenan Crane
//
#pragma once

#include <oz/math_util.h>

namespace oz {

    // w0, w1, w2, and w3 are the four cubic B-spline basis functions
    inline __host__ __device__ float w0(float a) { return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f); }
    inline __host__ __device__ float w1(float a) { return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f); }
    inline __host__ __device__ float w2(float a) { return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f); }
    inline __host__ __device__ float w3(float a) { return (1.0f/6.0f)*(a*a*a); }

    // g0 and g1 are the two amplitude functions
    inline __device__ float g0(float a) { return w0(a) + w1(a); }
    inline __device__ float g1(float a) { return w2(a) + w3(a); }

    // h0 and h1 are the two offset functions
    inline __device__ float h0(float a) { return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f; }
    inline __device__ float h1(float a) { return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f; }

    // filter 4 values using cubic splines
    template<class T> 
    __device__ T cubicFilter(float x, T c0, T c1, T c2, T c3) {
        T r;
        r = c0 * w0(x);
        r += c1 * w1(x);
        r += c2 * w2(x);
        r += c3 * w3(x);
        return r;
    }

    // slow but precise bicubic lookup using 16 texture lookups
    template<class T> 
    __device__ T tex2DBicubic(const texture<T, 2, cudaReadModeElementType> texref, float x, float y) {
        x -= 0.5f;
        y -= 0.5f;
        float px = floor(x);
        float py = floor(y);
        float fx = x - px;
        float fy = y - py;

        return cubicFilter<T>(
            fy,
            cubicFilter<T>(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
            cubicFilter<T>(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
            cubicFilter<T>(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
            cubicFilter<T>(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
        );
    }

    // fast bicubic texture lookup using 4 bilinear lookups
    template<class T>
    __device__ T tex2DFastBicubic(const texture<T, 2, cudaReadModeElementType> texref, float x, float y) {
        x -= 0.5f;
        y -= 0.5f;
        float px = floor(x);
        float py = floor(y);
        float fx = x - px;
        float fy = y - py;

        // note: we could store these functions in a lookup table texture, but maths is cheap
        float g0x = g0(fx);
        float g1x = g1(fx);
        float h0x = h0(fx);
        float h1x = h1(fx);
        float h0y = h0(fy);
        float h1y = h1(fy);

        T r = g0(fy) * ( g0x * tex2D(texref, px + h0x, py + h0y)   +
                         g1x * tex2D(texref, px + h1x, py + h0y) ) +
              g1(fy) * ( g0x * tex2D(texref, px + h0x, py + h1y)   +
                         g1x * tex2D(texref, px + h1x, py + h1y) );
        return r;
    }

    // higher-precision 2D bilinear lookup
    template<class T>  // return type, texture type
    __device__ T tex2DBilinear(const texture<T, 2, cudaReadModeElementType> tex, float x, float y) {
        x -= 0.5f;
        y -= 0.5f;
        float px = floorf(x);   // integer position
        float py = floorf(y);
        float fx = x - px;      // fractional position
        float fy = y - py;
        px += 0.5f;
        py += 0.5f;

        return ::lerp( ::lerp( tex2D(tex, px, py),        tex2D(tex, px + 1.0f, py), fx ),
                       ::lerp( tex2D(tex, px, py + 1.0f), tex2D(tex, px + 1.0f, py + 1.0f), fx ), fy );
    }

    // Catmull-Rom interpolation
    inline __host__ __device__ float catrom_w0(float a) { return a*(-0.5f + a*(1.0f - 0.5f*a)); }
    inline __host__ __device__ float catrom_w1(float a) { return 1.0f + a*a*(-2.5f + 1.5f*a); }
    inline __host__ __device__ float catrom_w2(float a) { return a*(0.5f + a*(2.0f - 1.5f*a)); }
    inline __host__ __device__ float catrom_w3(float a) { return a*a*(-0.5f + 0.5f*a); }

    template<class T>
    __device__ T catRomFilter(float x, T c0, T c1, T c2, T c3) {
        T r;
        r = c0 * catrom_w0(x);
        r += c1 * catrom_w1(x);
        r += c2 * catrom_w2(x);
        r += c3 * catrom_w3(x);
        return r;
    }

    // Note - can't use bilinear trick here because of negative lobes
    template<class T>
    __device__ T tex2DCatRom(const texture<T, 2, cudaReadModeElementType> texref, float x, float y) {
        x -= 0.5f;
        y -= 0.5f;
        float px = floor(x);
        float py = floor(y);
        float fx = x - px;
        float fy = y - py;

        return catRomFilter<T>(
            fy,
            catRomFilter<T>(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
            catRomFilter<T>(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
            catRomFilter<T>(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
            catRomFilter<T>(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
        );
    }

}
