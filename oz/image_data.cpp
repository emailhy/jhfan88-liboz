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
#include <oz/image_data.h>
#include <oz/gpu_cache.h>


oz::image_data::image_data() {
    format_ = FMT_INVALID;
    type_size_ = 0;
    ptr_ = 0;
    pitch_ = 0;
    w_ = h_ = 0;
}


oz::image_data::~image_data() {
}


oz::image_data_ex::image_data_ex() {
    ref_count_ = 1;
}


void oz::image_data_ex::add_ref() {
    ++ref_count_;
}


void oz::image_data_ex::release() {
    --ref_count_;
    if (ref_count_ == 0) delete this;
}


oz::image_data_cpu::image_data_cpu( unsigned w, unsigned h, oz::image_format_t format, unsigned type_size ) {
    if (format != FMT_STRUCT) {
        format_ = format;
        type_size_ = image_format_type_size(format);
    } else {
        format_ = (type_size != 0)? FMT_STRUCT : FMT_INVALID;
        type_size_ = type_size;
    }
    if (format != FMT_INVALID) {
        w_ = w;
        h_ = h;
        pitch_ = type_size_ * w_;
        ptr_ = ::malloc(pitch_ * h_);
        if (!ptr_) throw std::bad_alloc();
    } else {
        w_ = h_ = 0;
        pitch_ = 0;
        ptr_ = 0;
    }
}


oz::image_data_cpu::~image_data_cpu() {
    ::free(ptr_);
}


oz::image_data_gpu::image_data_gpu( unsigned w, unsigned h, oz::image_format_t format, unsigned type_size ) {
    if (format != FMT_STRUCT) {
        format_ = format;
        type_size_ = image_format_type_size(format);
    } else {
        format_ = (type_size != 0)? FMT_STRUCT : FMT_INVALID;
        type_size_ = type_size;
    }
    if (format != FMT_INVALID) {
        w_ = w;
        h_ = h;
        gpu_cache_alloc(&ptr_, &pitch_, type_size_ * w_, h_);
    } else {
        w_ = h_ = 0;
        pitch_ = 0;
        ptr_ = 0;
    }
}


oz::image_data_gpu::~image_data_gpu() {
    gpu_cache_free(ptr_, pitch_, type_size_ * w_, h_);
}
