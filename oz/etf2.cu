// by Jan Eric Kyprianidis <www.kyprianidis.com>
// based on a C++ implementation by Henry Kang <www.cs.umsl.edu/~kang>
#include <oz/etf2.h>
#include <oz/shuffle.h>
#include <oz/gpu_binder.h>
#include <oz/gpu_plm2.h>
#include <oz/launch_config.h>


texture<float,  2, cudaReadModeElementType> texSRC;
texture<float4, 2, cudaReadModeElementType> texSRC4;


__global__ void imp_etf_calc_grad( oz::gpu_plm2<float> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float2 t;
    t.x = (
           -1 * tex2D(texSRC, ix-1, iy-1) +
           -2 * tex2D(texSRC, ix-1, iy) + 
           -1 * tex2D(texSRC, ix-1, iy+1) +
            1 * tex2D(texSRC, ix+1, iy-1) +
            2 * tex2D(texSRC, ix+1, iy) + 
            1 * tex2D(texSRC, ix+1, iy+1)
           ) / 4.0f;

    t.y = (
           -1 * tex2D(texSRC, ix-1, iy-1) + 
           -2 * tex2D(texSRC, ix,   iy-1) + 
           -1 * tex2D(texSRC, ix+1, iy-1) +
            1 * tex2D(texSRC, ix-1, iy+1) +
            2 * tex2D(texSRC, ix,   iy+1) + 
            1 * tex2D(texSRC, ix+1, iy+1)
           ) / 4.0f;
    
    dst.write(ix, iy, length(t));
}


__global__ void imp_etf_calc_txtymag( oz::gpu_plm2<float3> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float2 t;
    t.x = (
           -1 * tex2D(texSRC, ix-1, iy-1) +
           -2 * tex2D(texSRC, ix-1, iy) + 
           -1 * tex2D(texSRC, ix-1, iy+1) +
            1 * tex2D(texSRC, ix+1, iy-1) +
            2 * tex2D(texSRC, ix+1, iy) + 
            1 * tex2D(texSRC, ix+1, iy+1)
           ) / 4.0f;

    t.y = (
           -1 * tex2D(texSRC, ix-1, iy-1) + 
           -2 * tex2D(texSRC, ix,   iy-1) + 
           -1 * tex2D(texSRC, ix+1, iy-1) +
            1 * tex2D(texSRC, ix-1, iy+1) +
            2 * tex2D(texSRC, ix,   iy+1) + 
            1 * tex2D(texSRC, ix+1, iy+1)
           ) / 4.0f;

    float len = length(t);
    if (len > 0)
        t /= len;

    len /= sqrtf(2); // hack to avoid normalization
    dst.write(ix, iy, make_float3(-t.y, t.x, len));
}


__global__ void imp_etf_smooth_x( oz::gpu_plm2<float3> dst, int half_w ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float3 p0 = make_float3(tex2D(texSRC4, ix, iy));

    float2 g = make_float2(0);
    for (int s = -half_w; s <= half_w; s++) {
        int x = ix+s; 
        int y = iy;
        if (x > dst.w-1) x = dst.w-1;
        else if (x < 0) x = 0;
        if (y > dst.h-1) y = dst.h-1;
        else if (y < 0) y = 0;
        float3 p = make_float3(tex2D(texSRC4, x, y));
        
        float factor = 1.0f;
        float angle = dot(make_float2(p0.x, p0.y), make_float2(p.x, p.y));
        if (angle < 0.0f) {
            factor = -1.0f; // reverse the direction
        }

        float mag_diff = p.z - p0.z; 
        float weight = mag_diff + 1; 

        g.x += weight * p.x * factor;
        g.y += weight * p.y * factor;
    }
               
    float len = length(g);
    if (len > 0)
        g /= len;

    dst.write(ix, iy, make_float3(g.x, g.y, p0.z));
}


__global__ void imp_etf_smooth_y( oz::gpu_plm2<float3> dst, int half_w ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float3 p0 = make_float3(tex2D(texSRC4, ix, iy));

    float2 g = make_float2(0);
    for (int t = -half_w; t <= half_w; t++) {
        int x = ix; 
        int y = iy+t;
        if (x > dst.w-1) x = dst.w-1;
        else if (x < 0) x = 0;
        if (y > dst.h-1) y = dst.h-1;
        else if (y < 0) y = 0;
        float3 p = make_float3(tex2D(texSRC4, x, y));
        
        float factor = 1.0f;
        float angle = dot(make_float2(p0.x, p0.y), make_float2(p.x, p.y));
        if (angle < 0.0f) {
            factor = -1.0f; // reverse the direction
        }

        float mag_diff = p.z - p0.z; 
        float weight = mag_diff + 1; 

        g.x += weight * p.x * factor;
        g.y += weight * p.y * factor;
    }
               
    float len = length(g);
    if (len > 0)
        g /= len;
    
    dst.write(ix, iy, make_float3(g.x, g.y, p0.z));
}


oz::gpu_image oz::etf_xy2( const gpu_image& src, int half_w, int M ) {
    launch_config cfg(src);

    gpu_image etf(src.size(), FMT_FLOAT3);
    {
        gpu_image tmp1(src.size(), FMT_FLOAT);
        gpu_binder<float> src_(texSRC, src);
        imp_etf_calc_grad<<<cfg.blocks(), cfg.threads()>>>(tmp1);
        OZ_CUDA_ERROR_CHECK();

        gpu_binder<float> tmp1_(texSRC, tmp1);
        imp_etf_calc_txtymag<<<cfg.blocks(), cfg.threads()>>>(etf);
        OZ_CUDA_ERROR_CHECK();
    }

    for (int k = 0; k < M; ++k) {
        gpu_image tmp3(src.size(), FMT_FLOAT3);
        
        gpu_binder<float3> etf_(texSRC4, etf);
        imp_etf_smooth_x<<<cfg.blocks(), cfg.threads()>>>(tmp3, half_w);
        OZ_CUDA_ERROR_CHECK();

        gpu_binder<float3> tmp3_(texSRC4, tmp3);
        imp_etf_smooth_y<<<cfg.blocks(), cfg.threads()>>>(etf, half_w);
        OZ_CUDA_ERROR_CHECK();
    }

    return shuffle(etf, 0, 1);
}
