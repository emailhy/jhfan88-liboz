// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include <oz/gpu_image.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_timer.h>
#include <cfloat>


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};


template <class T, unsigned int blockSize>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*2*blockSize + threadIdx.x;
    const unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    while (i < n) {
        mySum += g_idata[i];
        if (i + blockSize < n)
            mySum += g_idata[i+blockSize];
        i += gridSize;
    }

    sdata[tid] = mySum;
    __syncthreads();


    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}


template <class T, unsigned int nThreads, unsigned int nBlocks>
__global__ void
reduce7(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * nThreads + threadIdx.x;

    T mySum = 0;
    while (i < n) {
        mySum += g_idata[i];
        i += nThreads*nBlocks;
    }

    sdata[tid] = mySum;
    __syncthreads();

    if (nThreads >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (nThreads >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (nThreads >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        volatile T* smem = sdata;
        if (nThreads >=  64) { smem[tid] += smem[tid + 32]; }
        if (nThreads >=  32) { smem[tid] += smem[tid + 16]; }
        if (nThreads >=  16) { smem[tid] += smem[tid +  8]; }
        if (nThreads >=   8) { smem[tid] += smem[tid +  4]; }
        if (nThreads >=   4) { smem[tid] += smem[tid +  2]; }
        if (nThreads >=   2) { smem[tid] += smem[tid +  1]; }
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = threads * sizeof(float);

    switch (threads) {
    case 512:
        reduce6<float, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
        reduce7<float, 256, 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
        reduce6<float, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
        reduce6<float,  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 32:
        reduce6<float,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 16:
        reduce6<float,  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  8:
        reduce6<float,   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  4:
        reduce6<float,   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  2:
        reduce6<float,   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  1:
        reduce6<float,   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
}


template <unsigned nThreads, unsigned nBlocks>
__global__ void reduce8(float *dst, const float *g_idata, const unsigned w, const unsigned h, const unsigned pitch) {
    float *sdata = SharedMemory<float>();

    unsigned int tid = threadIdx.x;
    float mySum = 0;

    unsigned o = blockIdx.x * pitch;
    while (o < h*pitch) {
        unsigned int i = threadIdx.x;
        while (i < w) {
            mySum += g_idata[o+i];
            i += nThreads;
        }
        o += nBlocks*pitch;
    }

    sdata[tid] = mySum;
    __syncthreads();

    if (nThreads >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (nThreads >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (nThreads >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        volatile float* smem = sdata;
        if (nThreads >=  64) { smem[tid] += smem[tid + 32]; }
        if (nThreads >=  32) { smem[tid] += smem[tid + 16]; }
        if (nThreads >=  16) { smem[tid] += smem[tid +  8]; }
        if (nThreads >=   8) { smem[tid] += smem[tid +  4]; }
        if (nThreads >=   4) { smem[tid] += smem[tid +  2]; }
        if (nThreads >=   2) { smem[tid] += smem[tid +  1]; }
    }

    if (tid == 0)
        dst[blockIdx.x] = sdata[0];
}

//__device__ float4 get_float4(float4 const volatile & v) {
//        return make_float4(v.x, v.y, v.z, v.w);
//}


inline __device__ void minmax(float2 volatile& x, float2& a, float2 const volatile& b) {
    a = make_float2( fminf(a.x, b.x), fmaxf(a.y, b.y) );
    x.x = a.x;
    x.y = a.y;
}


template <unsigned nThreads, unsigned nBlocks>
__global__ void impl_minmax2(float2 *dst, const oz::gpu_plm2<float> src) {
    __shared__ float2 sdata[nThreads];

    unsigned int tmin = threadIdx.x;
    float2 myVal = make_float2(FLT_MAX, -FLT_MAX);

    unsigned o = blockIdx.x * src.stride;
    while (o < src.h * src.stride) {
        unsigned int i = threadIdx.x;
        while (i < src.w) {
            volatile float v = src.ptr[o+i];
            myVal = make_float2(fminf(myVal.x, v), fmaxf(myVal.y, v));
            i += nThreads;
        }
        o += nBlocks * src.stride;
    }

    sdata[tmin] = myVal;
    __syncthreads();

    if (nThreads >= 512) { if (tmin < 256) { minmax(sdata[tmin], myVal, sdata[tmin + 256]); } __syncthreads(); }
    if (nThreads >= 256) { if (tmin < 128) { minmax(sdata[tmin], myVal, sdata[tmin + 128]); } __syncthreads(); }
    if (nThreads >= 128) { if (tmin <  64) { minmax(sdata[tmin], myVal, sdata[tmin +  64]); } __syncthreads(); }

    if (tmin < 32) {
        volatile float2* smem = sdata;
        if (nThreads >=  64) { minmax(smem[tmin], myVal, smem[tmin + 32]); }
        if (nThreads >=  32) { minmax(smem[tmin], myVal, smem[tmin + 16]); }
        if (nThreads >=  16) { minmax(smem[tmin], myVal, smem[tmin +  8]); }
        if (nThreads >=   8) { minmax(smem[tmin], myVal, smem[tmin +  4]); }
        if (nThreads >=   4) { minmax(smem[tmin], myVal, smem[tmin +  2]); }
        if (nThreads >=   2) { minmax(smem[tmin], myVal, smem[tmin +  1]); }
    }

    if (tmin == 0) {
        dst[blockIdx.x] = sdata[0];
    }
}


void gpu_minmax2(const oz::gpu_image& src, float *pmin, float *pmax) {
    const unsigned nBlocks = 64;
    const unsigned nThreads = 128;

    static float2 *dst_gpu = 0;
    static float2 *dst_cpu = 0;
    if (!dst_cpu) {
        cudaMalloc(&dst_gpu, sizeof(float2)*nBlocks);
        cudaMallocHost(&dst_cpu, sizeof(float2)*nBlocks, cudaHostAllocPortable);
    }

    //unsigned smemSize = 2 * nThreads * sizeof(float);
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
    impl_minmax2<nThreads, nBlocks><<< dimGrid, dimBlock/*, smemSize */>>>(dst_gpu, oz::gpu_plm2<float>(src));
    cudaMemcpy(dst_cpu, dst_gpu, sizeof(float2)*nBlocks, cudaMemcpyDeviceToHost);

    if (pmin) {
        float m = dst_cpu[0].x;
        for (int i = 1; i < nBlocks; ++i) m = fminf(m, dst_cpu[i].x);
        *pmin = m;
    }
    if (pmax) {
        float m = dst_cpu[nBlocks].y;
        for (int i = 1; i < nBlocks; ++i) m = fmaxf(m, dst_cpu[i].y);
        *pmax = m;
    }
}
