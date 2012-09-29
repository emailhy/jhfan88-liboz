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
#include <oz/noise.h>
#include <oz/cpu_image.h>
#include <oz/generate.h>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <curand_kernel.h>


/*
    //polar form of the Box-Muller transformation
    static float second;
    static bool have_second = false;

    if (have_second) {
        return second;
    }

    float x1, x2, w, y1, y2;
    do {
        x1 = 2.0 * randf() - 1.0;
        x2 = 2.0 * randf() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ((w <= 0) || (w >= 1.0));

    w = sqrt( (-2.0 * ln( w ) ) / w );
    second = x2 * w;
    return x1 * w;
*/


static int randi() {
    static bool first = true;
    if (first) {
        first = false;
        std::srand(clock());
    }
    return std::rand();
}


static float randf() {
    return (float)randi()/RAND_MAX;
}


oz::gpu_image oz::noise_random( unsigned w, unsigned h, float a, float b ) {
    float *n = new float[w * h];
    float *p = n;
    float da = b - a;
	for (unsigned j = 0; j < h; ++j) {
        for (unsigned i = 0; i < w; ++i) {
            *p++ = a + da * randf();
        }
    }
    cpu_image dst(n, w*sizeof(float), w, h);
    delete[] n;
    return dst.gpu();
}


///////////////////////////////////////////////////////////////////////////////


namespace {
    static inline __device__ float actualNoise(float2 pos) {
        // This inliner is taken from: http://www.m3xbox.com/GPU_blog/?p=28
        float seed = 11.0f;
        return fract(sinf(dot( make_float2(cosf(pos.x), cosf(pos.y)) * sinf(seed),
            make_float2(15.12345f, 91.98765f))) * 115309.76543f);
        //return fract(sin(dot(pos, make_float2(12.9898f, 78.233f))) * 43758.5453f);
    }

    struct NoiseFast : public oz::generator<float> {
        float scale_;
        NoiseFast( float scale ) : scale_(scale) {}

        // by Holger Winnemoeller
        static inline __device__ float computeLinearNoise(float2 pos, float lScale) {
            // This changes the UNIFORM noise of the actualNoise function into GAUSSIAN NOISE
            // computes a scaled noise (lScale = local scale) which uses linear neighbor
            // interpolation (still artifacty, but better than nearest neighbor)
            float2 offset = make_float2(711.0f, 911.0f);
            pos = (pos+offset)/lScale;
            float2 newPos = floor(pos);
            float2 newRem = (pos-newPos);

            float n = actualNoise(newPos);
            float nr = actualNoise(newPos+make_float2(1,0));
            float nd = actualNoise(newPos+make_float2(0,1));
            float nrd = actualNoise(newPos+make_float2(1,1));

            float h1 = lerp(n,nr,newRem.x);
            float h2 = lerp(nd,nrd,newRem.x);

            return lerp(h1,h2,newRem.y);
        }

        inline __device__ float operator()( int ix, int iy ) const {
            return computeLinearNoise(make_float2(ix, iy), scale_);
        }
    };

    struct NoiseFast2 : public oz::generator<float> {
        float scale_, micro_;
        NoiseFast2( float scale, float micro ) : scale_(scale), micro_(micro) {}

        // by Holger Winnemoeller
        static inline __device__ float computeLinearNoise2(float2 pos, float lScale, float micro) {
            // This changes the UNIFORM noise of the actualNoise function into GAUSSIAN NOISE
            // computes a scaled noise (lScale = local scale) which uses linear neighbor
            // interpolation (still artifacty, but better than nearest neighbor)
            float2 offset = make_float2(711.0f, 911.0f);
            pos = (pos+offset)/lScale;
            float2 newPos = floor(pos);
            float2 newRem = (pos-newPos);

            float n = actualNoise(newPos);
            float nr = actualNoise(newPos+make_float2(1,0));
            float nd = actualNoise(newPos+make_float2(0,1));
            float nrd = actualNoise(newPos+make_float2(1,1));

            float h1 = lerp(n,nr,newRem.x);
            float h2 = lerp(nd,nrd,newRem.x);

            float v = lerp(h1,h2,newRem.y);
            v = (v < 0.5f)? 0 : 1; // smoothStep(noise_contrast,1.0-noise_contrast,v);
            return v + micro * actualNoise(pos);
        }

        inline __device__ float operator()( int ix, int iy ) const {
            return computeLinearNoise2(make_float2(ix, iy), scale_, micro_);
        }
    };
}


oz::gpu_image oz::noise_fast( unsigned w, unsigned h, float scale ) {
    return generate(w, h, NoiseFast(scale));
}


oz::gpu_image oz::noise_fast2( unsigned w, unsigned h, float scale, float micro ) {
    return generate(w, h, NoiseFast2(scale, micro));
}


///////////////////////////////////////////////////////////////////////////////


static const int NTHREADS = 64;
static const int NBLOCKS = 64;
__device__ curandState d_state[NTHREADS*NBLOCKS];


template <unsigned nBlocks, unsigned nThreads>
__global__ void rand_init() {
    int id = threadIdx .x + blockIdx.x * nThreads;
    curandState localState;
    curand_init(1234, id, 0, &localState);
    d_state[id] = localState;
}


static void rand_init() {
    static bool first = true;
    if (first) {
        first = false;
        rand_init<NBLOCKS,NTHREADS><<<NBLOCKS,NTHREADS>>>();
    }
}


template <unsigned nBlocks, unsigned nThreads>
__global__ void rand_generate_uniform( oz::gpu_plm2<float> dst, float a, float da ) {
    int id = threadIdx .x + blockIdx.x * nThreads;
    curandState localState = d_state[id];

    unsigned o = blockIdx.x * dst.stride;
    while (o < dst.h * dst.stride) {
        unsigned i = threadIdx.x;
        while (i < dst.w) {
            dst.ptr[o+i] = da * curand_uniform(&localState) + a;
            i += nThreads;
        }
        o += nBlocks * dst.stride;
    }

    d_state[id] = localState;
}


oz::gpu_image oz::noise_uniform( unsigned w, unsigned h, float a, float b ) {
	gpu_image dst(w, h, FMT_FLOAT);
    rand_init();
    rand_generate_uniform<NBLOCKS,NTHREADS><<<NBLOCKS,NTHREADS>>>(dst, a, b - a);
    return dst;
}


template <unsigned nBlocks, unsigned nThreads>
__global__ void rand_generate_normal( oz::gpu_plm2<float> dst, float mean, float stddev) {
    int id = threadIdx .x + blockIdx.x * nThreads;
    curandState localState = d_state[id];

    unsigned o = blockIdx.x * dst.stride;
    while (o < dst.h * dst.stride) {
        unsigned i = threadIdx.x;
        while (i < dst.w) {
            dst.ptr[o+i] = mean + stddev*curand_normal(&localState);
            i += nThreads;
        }
        o += nBlocks * dst.stride;
    }

    d_state[id] = localState;
}


oz::gpu_image oz::noise_normal( unsigned w, unsigned h, float mean, float variance ) {
    gpu_image dst(w, h, FMT_FLOAT);
    rand_init();
    rand_generate_normal<NBLOCKS,NTHREADS><<<NBLOCKS,NTHREADS>>>(dst, mean, sqrtf(variance));
    return dst;
}


template <unsigned nBlocks, unsigned nThreads, typename Op, typename T>
__global__ void impl_curand_op( oz::gpu_plm2<T> dst, const oz::gpu_plm2<T> src, Op op ) {
    int id = threadIdx .x + blockIdx.x * nThreads;
    curandState localState = d_state[id];

    unsigned os = blockIdx.x * src.stride;
    unsigned od = blockIdx.x * dst.stride;
    while (od < dst.h * dst.stride) {
        unsigned i = threadIdx.x;
        while (i < dst.w) {
            dst.write(od+i, op(src.read(os+i), &localState));
            i += nThreads;
        }
        os += nBlocks * src.stride;
        od += nBlocks * dst.stride;
    }

    d_state[id] = localState;
}


struct Gaussian {
    Gaussian(float mean, float variance) { m = mean; s = sqrtf(variance); }

    inline __device__ float operator()(float a, curandState *state) const {
        float x = curand_normal(state);
        float n = s * x + m;
        return a + n;
    }

    inline __device__ float3 operator()(float3 a, curandState *state) const {
        return make_float3( operator()(a.x, state), operator()(a.y, state), operator()(a.z, state));
    }

    float m;
    float s;
};


oz::gpu_image oz::add_gaussian_noise( const gpu_image& src, float mean, float variance ) {
    rand_init();
    Gaussian op(mean, variance);
    switch (src.format()) {
        case FMT_FLOAT:
            {
                gpu_image dst(src.size(), FMT_FLOAT);
                impl_curand_op<NBLOCKS,NTHREADS,Gaussian,float><<<NBLOCKS,NTHREADS>>>(dst, src, op);
                return dst;
            }
        case FMT_FLOAT3:
            {
                gpu_image dst(src.size(), FMT_FLOAT3);
                impl_curand_op<NBLOCKS,NTHREADS,Gaussian,float3><<<NBLOCKS,NTHREADS>>>(dst, src, op);
                return dst;
            }
        default:
            OZ_INVALID_FORMAT();
    }
}


struct SaltAndPepper {
    SaltAndPepper(float density) { d = density; }

    inline __device__ float operator()(float a, curandState *state) const {
        float x = curand_uniform(state);
        if (x < d/2) return 0;
        if ((x >= d/2) && (x < d)) return 1;
        return a;
    }

    inline __device__ float3 operator()(float3 a, curandState *state) const {
        return make_float3( operator()(a.x, state), operator()(a.y, state), operator()(a.z, state));
    }

    float d;
};


oz::gpu_image oz::add_salt_and_pepper_noise( const gpu_image& src, float density ) {
    rand_init();
    SaltAndPepper op(density);
    switch (src.format()) {
        case FMT_FLOAT:
            {
                gpu_image dst(src.size(), FMT_FLOAT);
                impl_curand_op<NBLOCKS,NTHREADS,SaltAndPepper,float><<<NBLOCKS,NTHREADS>>>(dst, src, op);
                return dst;
            }
        case FMT_FLOAT3:
            {
                gpu_image dst(src.size(), FMT_FLOAT3);
                impl_curand_op<NBLOCKS,NTHREADS,SaltAndPepper,float3><<<NBLOCKS,NTHREADS>>>(dst, src, op);
                return dst;
            }
        default:
            OZ_INVALID_FORMAT();
    }
}


struct Speckle {
    Speckle(float variance) { v = sqrtf(12*variance); }

    inline __device__ float operator()(float a, curandState *state) const {
        float x = curand_uniform(state);
        return a + v * a * (x - 0.5f);
    }

    inline __device__ float3 operator()(float3 a, curandState *state) const {
        return make_float3( operator()(a.x, state), operator()(a.y, state), operator()(a.z, state));
    }

    float v;
};


oz::gpu_image oz::add_speckle_noise( const gpu_image& src, float variance ) {
    rand_init();
    Speckle op(variance);
    switch (src.format()) {
        case FMT_FLOAT:
            {
                gpu_image dst(src.size(), FMT_FLOAT);
                impl_curand_op<NBLOCKS,NTHREADS,Speckle,float><<<NBLOCKS,NTHREADS>>>(dst, src, op);
                return dst;
            }
        case FMT_FLOAT3:
            {
                gpu_image dst(src.size(), FMT_FLOAT3);
                impl_curand_op<NBLOCKS,NTHREADS,Speckle,float3><<<NBLOCKS,NTHREADS>>>(dst, src, op);
                return dst;
            }
        default:
            OZ_INVALID_FORMAT();
    }
}
