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
#include <oz/cpu_image.h>
#include <oz/gpu_image.h>
#include <oz/convert.h>


template<typename T>
static void pack3( T *dst, unsigned dst_pitch,
                   const T* src, unsigned src_pitch, unsigned w, unsigned h )
{
    if (!dst_pitch) dst_pitch = 4*sizeof(T)*w;
    if (!src_pitch) src_pitch = 3*sizeof(T)*w;

    uchar *pp = (uchar*)dst;
    uchar *qq = (uchar*)src;
    for (unsigned j = 0; j < h; ++j) {
        T *p = (T*)pp;
        T *q = (T*)qq;
        for (unsigned i = 0; i < w; ++i) {
            *p++ = *q++;
            *p++ = *q++;
            *p++ = *q++;
            *p++ = 0;
        }
        pp += dst_pitch;
        qq += src_pitch;
    }
}


template<typename T>
static void unpack3( T *dst, unsigned dst_pitch,
                     const T* src, unsigned src_pitch, unsigned w, unsigned h )
{
    if (!dst_pitch) dst_pitch = 3*sizeof(T)*w;
    if (!src_pitch) src_pitch = 4*sizeof(T)*w;

    uchar *pp = (uchar*)dst;
    uchar *qq = (uchar*)src;
    for (unsigned j = 0; j < h; ++j) {
        T *p = (T*)pp;
        T *q = (T*)qq;
        for (unsigned i = 0; i < w; ++i) {
            *p++ = *q++;
            *p++ = *q++;
            *p++ = *q++;
            q++;
        }
        pp += dst_pitch;
        qq += src_pitch;
    }
}


oz::cpu_image::cpu_image( unsigned w, unsigned h, image_format_t format, unsigned type_size ) {
    d_ = new image_data_cpu(w, h, format, type_size);
}


oz::cpu_image::cpu_image( NppiSize size, image_format_t format, unsigned type_size ) {
    d_ = new image_data_cpu(size.width, size.height, format, type_size);
}


oz::cpu_image::cpu_image( const uchar *src, unsigned src_pitch, unsigned w, unsigned h ) {
    d_ = new image_data_cpu(w, h, FMT_UCHAR);
    cudaMemcpy2D(ptr(), pitch(),
                 src, src_pitch? src_pitch : sizeof(uchar)*w, sizeof(uchar)*w, h,
                 cudaMemcpyHostToHost);
}


oz::cpu_image::cpu_image( const uchar2 *src, unsigned src_pitch, unsigned w, unsigned h ) {
    d_ = new image_data_cpu(w, h, FMT_UCHAR2);
    cudaMemcpy2D(ptr(), pitch(),
                 src, src_pitch? src_pitch : sizeof(uchar2)*w, sizeof(uchar2)*w, h,
                 cudaMemcpyHostToHost);
}


oz::cpu_image::cpu_image( const uchar3 *src, unsigned src_pitch, unsigned w, unsigned h ) {
    d_ = new image_data_cpu(w, h, FMT_UCHAR3);
    pack3<uchar>((uchar*)ptr(), pitch(), (uchar*)src, src_pitch, w, h);
}


oz::cpu_image::cpu_image( const uchar4 *src, unsigned src_pitch, unsigned w, unsigned h, bool ignore_alpha )
{
    d_ = new image_data_cpu(w, h, ignore_alpha? FMT_UCHAR3 : FMT_UCHAR4);
    cudaMemcpy2D(ptr(), pitch(),
                 src, src_pitch? src_pitch : sizeof(uchar4)*w, sizeof(uchar4)*w, h,
                 cudaMemcpyHostToHost);
}


oz::cpu_image::cpu_image( const float *src, unsigned src_pitch, unsigned w, unsigned h ) {
    d_ = new image_data_cpu(w, h, FMT_FLOAT);
    cudaMemcpy2D(ptr(), pitch(),
                 src, src_pitch? src_pitch : sizeof(float)*w, sizeof(float)*w, h,
                 cudaMemcpyHostToHost);
}


oz::cpu_image::cpu_image( const float2 *src, unsigned src_pitch, unsigned w, unsigned h ) {
    d_ = new image_data_cpu(w, h, FMT_FLOAT2);
    cudaMemcpy2D(d_->ptr(), d_->pitch(),
                 src, src_pitch? src_pitch : sizeof(float2)*w, sizeof(float2)*w, h,
                 cudaMemcpyHostToHost);
}


oz::cpu_image::cpu_image( const float3 *src, unsigned src_pitch, unsigned w, unsigned h ) {
    d_ = new image_data_cpu(w, h, FMT_FLOAT3);
    pack3<float>((float*)ptr(), pitch(), (float*)src, src_pitch, w, h);
}


oz::cpu_image::cpu_image( const float4 *src, unsigned src_pitch, unsigned w, unsigned h, bool ignore_alpha )
{
    d_ = new image_data_cpu(w, h, ignore_alpha? FMT_FLOAT3 : FMT_FLOAT4);
    cudaMemcpy2D(ptr(), pitch(),
                 src, src_pitch? src_pitch : sizeof(float4)*w, sizeof(float4)*w, h,
                 cudaMemcpyHostToHost);
}


oz::cpu_image::cpu_image( const gpu_image& img ) : d_(0) {
    operator=(img.cpu());
}


const oz::cpu_image& oz::cpu_image::operator=( const gpu_image& img ) {
    return operator=(img.cpu());
}


void oz::cpu_image::clear() {
    if (is_valid()) {
        memset(ptr(), 0, pitch()*h());
    }
}


oz::cpu_image oz::cpu_image::clone() const {
    if (!is_valid()) return cpu_image();
    cpu_image dst(size(), format(), type_size());
    cudaMemcpy2D(dst.ptr<void>(), dst.pitch(), ptr<void>(), pitch(),
                 row_size(), h(), cudaMemcpyHostToHost);
    return dst;
}


oz::gpu_image oz::cpu_image::gpu() const {
    if (!is_valid()) return gpu_image();
    gpu_image dst(size(), format(), type_size());
    cudaMemcpy2D(dst.ptr<void>(), dst.pitch(), ptr<void>(), pitch(),
                 row_size(), h(), cudaMemcpyHostToDevice);
    return dst;
}


oz::cpu_image oz::cpu_image::convert( image_format_t format, bool clone ) const {
    return oz::convert(*this, format, clone);
}


oz::cpu_image oz::cpu_image::copy( int x1, int y1, int x2, int y2 ) const {
    int cw = x2 - x1 + 1;
    int ch = y2 - y1 + 1;
    if ((x1 < 0) || (x2 >= (int)w()) ||
        (y1 < 0) || (y2 >= (int)h()) ||
        (cw <= 0) || (ch <= 0)) OZ_X() << "Invalid region!";

    cpu_image dst(cw, ch, format());
    uchar *src_ptr = ptr<uchar>() + y1 * row_size() + x1 * type_size();
    cudaMemcpy2D(dst.ptr(), dst.pitch(), src_ptr, pitch(), dst.row_size(), dst.h(), cudaMemcpyHostToHost);
    return dst;
}


void oz::cpu_image::get( uchar *dst, unsigned dst_pitch ) const {
    OZ_CHECK_FORMAT(format(), FMT_UCHAR);
    cudaMemcpy2D(dst, dst_pitch? dst_pitch : sizeof(uchar)*w(),
                 ptr(), pitch(), sizeof(uchar)*w(), h(), cudaMemcpyHostToHost);
}


void oz::cpu_image::get( uchar2 *dst, unsigned dst_pitch ) const {
    OZ_CHECK_FORMAT(format(), FMT_UCHAR2);
    cudaMemcpy2D(dst, dst_pitch? dst_pitch : sizeof(uchar2)*w(),
                 ptr(), pitch(), sizeof(uchar2)*w(), h(), cudaMemcpyHostToHost);
}


void oz::cpu_image::get( uchar3 *dst, unsigned dst_pitch ) const {
    OZ_CHECK_FORMAT(format(), FMT_UCHAR3);
    unpack3<uchar>((uchar*)dst, dst_pitch, (uchar*)ptr(), pitch(), w(), h());
}


void oz::cpu_image::get( uchar4 *dst, unsigned dst_pitch ) const {
    OZ_CHECK_FORMAT2(format(), FMT_UCHAR3, FMT_UCHAR4);
    cudaMemcpy2D(dst, dst_pitch? dst_pitch : sizeof(uchar4)*w(),
                 ptr(), pitch(), sizeof(uchar4)*w(), h(), cudaMemcpyHostToHost);
}


void oz::cpu_image::get( float *dst, unsigned dst_pitch ) const {
    OZ_CHECK_FORMAT(format(), FMT_FLOAT);
    cudaMemcpy2D(dst, dst_pitch? dst_pitch : sizeof(float)*w(),
                 ptr(), pitch(), sizeof(float)*w(), h(), cudaMemcpyHostToHost);
}


void oz::cpu_image::get( float2 *dst, unsigned dst_pitch ) const {
    OZ_CHECK_FORMAT(format(), FMT_FLOAT2);
    cudaMemcpy2D(dst, dst_pitch? dst_pitch : sizeof(float2)*w(),
                 ptr(), pitch(), sizeof(float2)*w(), h(), cudaMemcpyHostToHost);
}


void oz::cpu_image::get( float3 *dst, unsigned dst_pitch ) const {
    OZ_CHECK_FORMAT(format(), FMT_FLOAT3);
    unpack3<float>((float*)dst, dst_pitch, (float*)ptr(), pitch(), w(), h());
}


void oz::cpu_image::get( float4 *dst, unsigned dst_pitch ) const {
    OZ_CHECK_FORMAT2(format(), FMT_FLOAT3, FMT_FLOAT4);
    cudaMemcpy2D(dst, dst_pitch? dst_pitch : sizeof(float4)*w(),
                 ptr(), pitch(), sizeof(float4)*w(), h(), cudaMemcpyHostToHost);
}
