// by Jan Eric Kyprianidis <www.kyprianidis.com>
// based on a C++ implementation by Henry Kang <www.cs.umsl.edu/~kang>
#include <oz/ssia.h>
#include <oz/etf2.h>
#include <oz/color.h>
#include <oz/dog.h>
#include <oz/launch_config.h>
#include <oz/gpu_binder.h>
#include <oz/gpu_plm2.h>


texture<float4,  2, cudaReadModeElementType> texSRC;
texture<float4,  2, cudaReadModeElementType> texIx;
texture<float4,  2, cudaReadModeElementType> texIy;
texture<float4,  2, cudaReadModeElementType> texIxx;
texture<float4,  2, cudaReadModeElementType> texIxy;
texture<float4,  2, cudaReadModeElementType> texIyy;
texture<float2,  2, cudaReadModeElementType> texETF;
texture<float,  2, cudaReadModeElementType> texDOG;
const int half1 = 1;


__global__ void imp_cmcf_IxIy( oz::gpu_plm2<float3> Ix, oz::gpu_plm2<float3> Iy ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= Ix.w || iy >= Ix.h) 
        return;

    float3 c = make_float3(tex2D(texSRC, ix, iy));

    float3 r, l;
    if (ix < Ix.w-1) r = make_float3(tex2D(texSRC, ix+1, iy)); else r = c;
    if (ix > 0)   l = make_float3(tex2D(texSRC, ix-1, iy)); else l = c;
    Ix.write(ix, iy, (r - l) / 2); 

    float3 t, b;
    if (iy < Iy.h-1) t = make_float3(tex2D(texSRC, ix, iy+1)); else t = c;
    if (iy > 0)   b = make_float3(tex2D(texSRC, ix, iy-1)); else b = c;
    Iy.write(ix, iy, (t - b) / 2);
}


__global__ void imp_cmcf_IxxIxyIyy( oz::gpu_plm2<float3> Ixx, oz::gpu_plm2<float3> Ixy, oz::gpu_plm2<float3> Iyy ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= Ixx.w || iy >= Ixx.h) 
        return;

    float3 c = make_float3(tex2D(texIx, ix, iy));
    float3 r, l;
    if (ix < Ixx.w-1) r = make_float3(tex2D(texIx, ix+1, iy)); else r = c;
    if (ix > 0)   l = make_float3(tex2D(texIx, ix-1, iy)); else l = c;
    Ixx.write(ix, iy, (r - l) / 2); 

    float3 t, b;
    if (iy < Ixx.h-1) t = make_float3(tex2D(texIx, ix, iy+1)); else t = c;
    if (iy > 0)   b = make_float3(tex2D(texIx, ix, iy-1)); else b = c;
    Ixy.write(ix, iy, (t - b) / 2);

    c = make_float3(tex2D(texIy, ix, iy));
    if (iy < Ixx.h-1) t = make_float3(tex2D(texIy, ix, iy+1)); else t = c;
    if (iy > 0)   b = make_float3(tex2D(texIy, ix, iy-1)); else b = c;
    Iyy.write(ix, iy, (t - b) / 2);
}


__global__ void imp_cmcf_main( oz::gpu_plm2<float3> dst, float r ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;
            
    float3 val = make_float3(tex2D(texSRC, ix, iy));

    float3 Ix = make_float3(tex2D(texIx, ix, iy));
    float3 Iy = make_float3(tex2D(texIy, ix, iy));
    float3 Ixx = make_float3(tex2D(texIxx, ix, iy));
    float3 Ixy = make_float3(tex2D(texIxy, ix, iy));
    float3 Iyy = make_float3(tex2D(texIyy, ix, iy));
    float2 t = tex2D(texETF, ix, iy);
    t = make_float2(t.y, -t.x);

    float3 diff_D = Ixx*Iy*Iy - 2*Ix*Iy*Ixy + Iyy*Ix*Ix;
    float3 diff_N = Ix*Ix + Iy*Iy;

    if (diff_N.x > 0) {
        float2 g = normalize(make_float2(Ix.x, Iy.x)); 
        float speed = (1 - r) + r * abs(dot(t, g)); 
        val.x += speed * diff_D.x / diff_N.x;
    }
    if (diff_N.y > 0) {
        float2 g = normalize(make_float2(Ix.y, Iy.y)); 
        float speed = (1 - r) + r * abs(dot(t, g)); 
        val.y += speed * diff_D.y / diff_N.y;
    }
    if (diff_N.z > 0) {
        float2 g = normalize(make_float2(Ix.z, Iy.z)); 
        float speed = (1 - r) + r * abs(dot(t, g)); 
        val.z += speed * diff_D.z / diff_N.z;
    }

    dst.write(ix, iy, val);
}


__global__ void imp_shock_x( oz::gpu_plm2<float3> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    int sign = tex2D(texDOG, ix, iy) > 0;

    float max_sum = -1;
    float min_sum = 1000000;
    float3 max_val;
    float3 min_val;
            
    for (int s = -half1; s <= half1; s++) {
        ////////////////////////
        int x = ix + s; 
        int y = iy;
        /////////////////////////////////////////////////////
        if (x > dst.w-1 || x < 0 || y > dst.h-1 || y < 0) 
            continue;
        /////////////////////////////////////////////////////
        if (sign == (tex2D(texDOG, x, y) > 0)) {
            float3 c = make_float3(tex2D(texSRC, x, y));
            float sum = c.x + c.y + c.z;
            if (sum > max_sum) { 
                max_sum = sum; 
                max_val = c;
            }
            if (sum < min_sum) { 
                min_sum = sum; 
                min_val = c;
            }
        }
    }

    dst.write(ix, iy, sign? max_val : min_val);
}


__global__ void imp_shock_y( oz::gpu_plm2<float3> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;
                                       
    int sign = tex2D(texDOG, ix, iy) > 0;

    float max_sum = -1;
    float min_sum = 1000000;
    float3 max_val;
    float3 min_val;
            
    for (int t = -half1; t <= half1; t++) {
        ////////////////////////
        int x = ix; 
        int y = iy + t;
        /////////////////////////////////////////////////////
        if (x > dst.w-1 || x < 0 || y > dst.h-1 || y < 0) 
            continue;
        ///////////////////////////////////////////////////// 
        if (sign == (tex2D(texDOG, x, y) > 0)) {
            float3 c = make_float3(tex2D(texSRC, x, y));
            float sum = c.x + c.y + c.z;
            if (sum > max_sum) { 
                max_sum = sum; 
                max_val = c;
            }
            if (sum < min_sum) { 
                min_sum = sum; 
                min_val = c;
            }
        }
    }

    dst.write(ix, iy, sign? max_val : min_val);
}


oz::gpu_image oz::ssia_cmcf( const gpu_image& src, const gpu_image& etf, float weight ) {
    launch_config cfg(src);
    
    gpu_image Ix(src.size(), FMT_FLOAT3);
    gpu_image Iy(src.size(), FMT_FLOAT3);
    gpu_image Ixx(src.size(), FMT_FLOAT3);
    gpu_image Ixy(src.size(), FMT_FLOAT3);
    gpu_image Iyy(src.size(), FMT_FLOAT3);

    gpu_binder<float3> src_(texSRC, src);
    imp_cmcf_IxIy<<<cfg.blocks(), cfg.threads()>>>( Ix, Iy );
    OZ_CUDA_ERROR_CHECK();

    gpu_binder<float3> Ix_(texIx, Ix);
    gpu_binder<float3> Iy_(texIy, Iy);
    imp_cmcf_IxxIxyIyy<<<cfg.blocks(), cfg.threads()>>>( Ixx, Ixy, Iyy );
    OZ_CUDA_ERROR_CHECK();

    gpu_image dst(src.size(), FMT_FLOAT3);
    gpu_binder<float3> Ixx_(texIxx, Ixx);
    gpu_binder<float3> Ixy_(texIxy, Ixy);
    gpu_binder<float3> Iyy_(texIyy, Iyy);
    gpu_binder<float2> ETF_(texETF, etf);
    imp_cmcf_main<<<cfg.blocks(), cfg.threads()>>>(dst, weight);
    OZ_CUDA_ERROR_CHECK();

    return dst;
}


oz::gpu_image oz::ssia_shock( const gpu_image& src, const gpu_image& dog ) {
    launch_config cfg(src);

    gpu_binder<float> dog_(texDOG, dog);

    gpu_image tmp(src.size(), FMT_FLOAT3);
    {
        gpu_binder<float3> src_(texSRC, src);
        imp_shock_x<<<cfg.blocks(), cfg.threads()>>>(tmp);
        OZ_CUDA_ERROR_CHECK();
    }

    gpu_image dst(src.size(), FMT_FLOAT3);
    {
        gpu_binder<float3> tmp_(texSRC, tmp);
        imp_shock_y<<<cfg.blocks(), cfg.threads()>>>(dst);
        OZ_CUDA_ERROR_CHECK();
    }

    return dst;
}


oz::gpu_image oz::ssia( const gpu_image& src, int total_N, int cmcf_N, float cmcf_weight,
                        int etf_N, int etf_halfw, float shock_sigma, float shock_tau )
{
    gpu_image gray;
    gpu_image dog;

    gpu_image img = src;
    for (int l = 0; l < total_N; ++l) {
        gray = rgb2gray(img);
        gpu_image etf = etf_xy2(gray, etf_halfw, etf_N);
        for (int k = 0; k < cmcf_N; ++k) {
            img = ssia_cmcf(img, etf, cmcf_weight);
        }

        if (shock_sigma > 0) {
            gray = rgb2gray(img);
            dog = dog_filter(gray, shock_sigma, shock_sigma * 1.6f, shock_tau, 0);
            img = ssia_shock(img, dog);
        }
    }

    if (shock_sigma > 0) {
        gray = rgb2gray(img);
        dog = dog_filter(gray, 1.5f, 1.5f * 1.6f, shock_tau, 0);
        img = ssia_shock(img, dog);
    }

    return img;
}
