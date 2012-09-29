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
#include <oz/gpu_image.h>
#include <oz/cpu_image.h>
#include <oz/convert.h>


oz::gpu_image::gpu_image( unsigned w, unsigned h, image_format_t format, unsigned type_size ) {
    d_ = new image_data_gpu(w, h, format, type_size);
}


oz::gpu_image::gpu_image( NppiSize size, image_format_t format, unsigned type_size ) {
    d_ = new image_data_gpu(size.width, size.height, format, type_size);
}


oz::gpu_image::gpu_image( unsigned w, unsigned h, float value ) {
    d_ = new image_data_gpu(w, h, FMT_FLOAT);
    fill(value, 0, 0, w, h);
}


oz::gpu_image::gpu_image( unsigned w, unsigned h, float2 value ) {
    d_ = new image_data_gpu(w, h, FMT_FLOAT2);
    fill(value, 0, 0, w, h);
}


oz::gpu_image::gpu_image( unsigned w, unsigned h, float3 value ) {
    d_ = new image_data_gpu(w, h, FMT_FLOAT3);
    fill(value, 0, 0, w, h);
}


oz::gpu_image::gpu_image( unsigned w, unsigned h, float4 value ) {
    d_ = new image_data_gpu(w, h, FMT_FLOAT4);
    fill(value, 0, 0, w, h);
}


oz::gpu_image::gpu_image( const cpu_image& img ) : d_(0) {
    operator=(img.gpu());
}


const oz::gpu_image& oz::gpu_image::operator=( const cpu_image& img ) {
    return operator=(img.gpu());
}


void oz::gpu_image::clear() {
    OZ_CUDA_SAFE_CALL(cudaMemset2D(ptr(), pitch(), 0, row_size(), h()));
}


void oz::gpu_image::clear_white() {
    switch (format()) {
        case FMT_FLOAT:
            fill(1, 0, 0, w(), h());
            break;
        case FMT_FLOAT2:
            fill(make_float2(1), 0, 0, w(), h());
            break;
        case FMT_FLOAT3:
            fill(make_float3(1), 0, 0, w(), h());
            break;
        case FMT_FLOAT4:
            fill(make_float4(1), 0, 0, w(), h());
            break;
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::gpu_image::clone() const {
    if (!is_valid()) return gpu_image();
    gpu_image dst(size(), format(), type_size());
    OZ_CUDA_SAFE_CALL(cudaMemcpy2D(dst.ptr<void>(), dst.pitch(), ptr<void>(), pitch(),
                                   row_size(), h(), cudaMemcpyDeviceToDevice));
    return dst;
}


oz::cpu_image oz::gpu_image::cpu() const {
    if (!is_valid()) return cpu_image();
    cpu_image dst(size(), format(), type_size());
    OZ_CUDA_SAFE_CALL(cudaMemcpy2D(dst.ptr<void>(), dst.pitch(), ptr<void>(), pitch(),
                                   row_size(), h(), cudaMemcpyDeviceToHost));
    return dst;
}


oz::gpu_image oz::gpu_image::convert( image_format_t format, bool clone ) const {
    return oz::convert(*this, format, clone);
}

