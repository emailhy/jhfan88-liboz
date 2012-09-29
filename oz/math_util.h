/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 * 
 */

/*
   This file implements common mathematical operations on vector types
   (float3, float4 etc.) since these are not provided as standard by CUDA.

   The syntax is modelled on the Cg standard library.
*/

#pragma once

#include <oz/config.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <nppdefs.h>

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef __CUDACC__
#include <cmath>

inline float fminf(float a, float b) { return a < b ? a : b; }
inline float fmaxf(float a, float b) { return a > b ? a : b; }

inline int max(int a, int b) { return a > b ? a : b; }
inline int min(int a, int b) { return a < b ? a : b; }

inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }

#ifdef _MSC_VER
inline float copysignf(float x, float y) { return (float)_copysign(x, y); }
#endif

#endif

// float functions
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t) { return a + t*(b-a); }
inline __device__ __host__ float clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }

// int2 functions
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ int2 operator-(int2 &a) { return make_int2(-a.x, -a.y); }
inline __host__ __device__ int2 operator+(int2 a, int2 b) { return make_int2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ void operator+=(int2 &a, int2 b) { a.x += b.x; a.y += b.y; }
inline __host__ __device__ int2 operator-(int2 a, int2 b) { return make_int2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ void operator-=(int2 &a, int2 b) { a.x -= b.x; a.y -= b.y; }
inline __host__ __device__ int2 operator*(int2 a, int2 b) { return make_int2(a.x * b.x, a.y * b.y); }
inline __host__ __device__ int2 operator*(int2 a, int s) { return make_int2(a.x * s, a.y * s); }
inline __host__ __device__ int2 operator*(int s, int2 a) { return make_int2(a.x * s, a.y * s); }
inline __host__ __device__ void operator*=(int2 &a, int s) { a.x *= s; a.y *= s; }

// float2 functions
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s) { return make_float2(s, s); }
inline __host__ __device__ float2 make_float2(int2 a) { return make_float2(float(a.x), float(a.y)); }
inline __host__ __device__ float2 operator-(float2 &a) { return make_float2(-a.x, -a.y); }
inline __host__ __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ void operator+=(float2 &a, float2 b) { a.x += b.x; a.y += b.y; }
inline __host__ __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ void operator-=(float2 &a, float2 b) { a.x -= b.x; a.y -= b.y; }
inline __host__ __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __host__ __device__ float2 operator*(float2 a, float s) { return make_float2(a.x * s, a.y * s); }
inline __host__ __device__ float2 operator*(float s, float2 a) { return make_float2(a.x * s, a.y * s); }
inline __host__ __device__ void operator*=(float2 &a, float s) { a.x *= s; a.y *= s; }
inline __host__ __device__ float2 operator/(float2 a, float2 b) { return make_float2(a.x / b.x, a.y / b.y); }
inline __host__ __device__ float2 operator/(float2 a, float s) { float inv = 1.0f / s; return a * inv; }
inline __host__ __device__ float2 operator/(float s, float2 a) { float inv = 1.0f / s; return a * inv; }
inline __host__ __device__ void operator/=(float2 &a, float s) { float inv = 1.0f / s; a *= inv; }
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t) { return a + t*(b-a); }
inline __device__ __host__ float2 clamp(float2 v, float a, float b) { return make_float2(clamp(v.x, a, b), clamp(v.y, a, b)); }
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b) { return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
inline __host__ __device__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
inline __host__ __device__ float length(float2 v) { return sqrtf(dot(v, v)); }
inline __host__ __device__ float2 normalize(float2 v) { float invLen = rsqrtf(dot(v, v)); return v * invLen; }
inline __host__ __device__ float2 floor(const float2 v) { return make_float2(floor(v.x), floor(v.y)); }
inline __host__ __device__ float2 reflect(float2 i, float2 n) { return i - 2.0f * n * dot(n,i); }
inline __host__ __device__ float2 fabs(float2 v) { return make_float2(fabs(v.x), fabs(v.y)); }

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);  // discards w
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

// negate
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// min
static __inline__ __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

// max
static __inline__ __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// addition
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float3 floor(const float3 v)
{
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

// reflect
inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

// absolute value
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

// float4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

// negate
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// min
static __inline__ __host__ __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

// max
static __inline__ __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

// addition
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// subtract
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

// multiply
inline __host__ __device__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ void operator*=(float4 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

// divide
inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ float4 operator/(float4 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float4 operator/(float s, float4 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float4 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

// dot product
inline __host__ __device__ float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// length
inline __host__ __device__ float length(float4 r)
{
    return sqrtf(dot(r, r));
}

// normalize
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float4 floor(const float4 v)
{
    return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

// absolute value
inline __host__ __device__ float4 fabs(float4 v)
{
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

// int3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

// negate
inline __host__ __device__ int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}

// min
inline __host__ __device__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

// max
inline __host__ __device__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// addition
inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ int3 operator*(int3 a, int s)
{
    return make_int3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ int3 operator*(int s, int3 a)
{
    return make_int3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(int3 &a, int s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ int3 operator/(int3 a, int3 b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ int3 operator/(int3 a, int s)
{
    return make_int3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ int3 operator/(int s, int3 a)
{
    return make_int3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ void operator/=(int3 &a, int s)
{
    a.x /= s; a.y /= s; a.z /= s;
}

// clamp
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}


// uint3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(float3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

// min
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

// max
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// addition
inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ uint3 operator*(uint3 a, uint s)
{
    return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ uint3 operator*(uint s, uint3 a)
{
    return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(uint3 &a, uint s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ uint3 operator/(uint3 a, uint3 b)
{
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ uint3 operator/(uint3 a, uint s)
{
    return make_uint3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ uint3 operator/(uint s, uint3 a)
{
    return make_uint3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ void operator/=(uint3 &a, uint s)
{
    a.x /= s; a.y /= s; a.z /= s;
}

// clamp
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}


////////////////////////////////////////////////////////////////////////////////
// by Jan Eric Kyprianidis


template<typename T> T make_zero();
template<> inline __host__ __device__ float  make_zero() { return 0; }
template<> inline __host__ __device__ float2 make_zero() { return make_float2(0); }
template<> inline __host__ __device__ float3 make_zero() { return make_float3(0); }
template<> inline __host__ __device__ float4 make_zero() { return make_float4(0); }


template<typename T, typename V> inline __host__ __device__ T make_T(V v) { return v; };
template<> inline __host__ __device__ uchar3  make_T(uchar4 v) { return make_uchar3(v.x, v.y, v.z); }
template<> inline __host__ __device__ uchar4  make_T(uchar3 v) { return make_uchar4(v.x, v.y, v.z, 1); }
template<> inline __host__ __device__ float3  make_T(float4 v) { return make_float3(v); }
template<> inline __host__ __device__ float4  make_T(float3 v) { return make_float4(v, 1); }


inline __host__ __device__ float sum(float  v) { return v; }
inline __host__ __device__ float sum(float2 v) { return v.x + v.y; }
inline __host__ __device__ float sum(float3 v) { return v.x + v.y + v.z; }
inline __host__ __device__ float sum(float4 v) { return v.x + v.y + v.z + v.w; }

inline __host__ __device__ float sum_sqrtf( float  v ) { return sqrtf(v); }
inline __host__ __device__ float sum_sqrtf( float2 v ) { return sqrtf(v.x) + sqrtf(v.y); }
inline __host__ __device__ float sum_sqrtf( float3 v ) { return sqrtf(v.x) + sqrtf(v.y) + sqrtf(v.z); }
inline __host__ __device__ float sum_sqrtf( float4 v ) { return sqrtf(v.x) + sqrtf(v.y) + sqrtf(v.z) + sqrtf(v.w); }


inline __host__ __device__ float2 make_float2(float4 a) {
    return make_float2(a.x, a.y);  // discards z, w
}


inline __host__ __device__ float2 rotate(float2 a, float angle) { 
    float c = cosf(angle);
    float s = sinf(angle);
    return make_float2( c * a.x - s * a.y, s * a.x + c *a.y );
}


inline __host__ __device__ bool operator==(NppiSize a, NppiSize b) {
    return (a.width == b.width) || (a.height == b.height);
}


inline __host__ __device__ bool operator!=(NppiSize a, NppiSize b) {
    return (a.width != b.width) || (a.height != b.height);
}


inline __host__ __device__ bool operator==(uint2 a, uint2 b) {
    return (a.x == b.x) && (a.y == b.y);
}


inline __host__ __device__ bool operator!=(uint2 a, uint2 b) {
    return (a.x != b.x) || (a.y != b.y);
}


inline __host__ __device__ float dot(float a, float b) { 
    return a*b; 
}


inline __host__ __device__ float length(float a) { 
    return fabsf(a); 
}


template<typename T>
inline __host__ __device__ float squared(T x) { 
    return dot(x, x); 
}


inline __host__ __device__ float fract(float x)   { return x - floor(x); }
inline __host__ __device__ float2 fract(float2 x) { return x - floor(x); }
inline __host__ __device__ float3 fract(float3 x) { return x - floor(x); }
inline __host__ __device__ float4 fract(float4 x) { return x - floor(x); }


inline __host__ __device__ float sign(float x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}


inline __host__ __device__ float smoothstep(float a, float b, float x) {
    float t = (x - a) / (b - a);
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    return t * t * (3 - 2 * t);
}


inline __host__ __device__ float4 operator*(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}


inline __host__ __device__ float radians(float deg) { 
    return static_cast<float>(deg * CUDART_PI_F / 180.0f ); 
}
    

inline __host__ __device__ float degrees(float rad) { 
    return static_cast<float>(rad * 180.0f / CUDART_PI_F); 
}


namespace oz {
    using ::abs;
    using ::clamp;
    using ::lerp;
    using ::max;
    using ::min;
    using ::clamp;
}