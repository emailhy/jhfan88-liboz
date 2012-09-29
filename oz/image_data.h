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
#pragma once

#include <cassert>
#include <cmath>
#include <oz/image_format.h>
#include <oz/math_util.h>
#include <oz/exception.h>

namespace oz {

    class OZAPI image_data {
    public:
        image_data();
        virtual ~image_data();

        virtual void add_ref() = 0;
        virtual void release() = 0;

        image_format_t format() const { return format_; }
        unsigned type_size() const { return type_size_; }
        void* ptr() const { return ptr_; }
        unsigned pitch() const { return pitch_; }
        unsigned row_size() const { return type_size_ * w_; }
        unsigned padding() const { return pitch_ - type_size_ * w_; }
        unsigned stride() const { assert(pitch_ % type_size_ == 0); return pitch_ / type_size_; }
        unsigned w() const { return w_; }
        unsigned h() const { return h_; }
        unsigned N() const { return w_ * h_; }
        NppiSize size() const { NppiSize s = { w_, h_ }; return s; }

    protected:
        image_format_t format_;
        unsigned type_size_;
        void *ptr_;
        unsigned pitch_;
        unsigned w_;
        unsigned h_;

    private:
        image_data(const image_data&);
        const image_data& operator=(const image_data&);
    };


    class OZAPI image_data_ex : public image_data {
    public:
        image_data_ex();
        virtual void add_ref();
        virtual void release();
    private:
        int ref_count_;
    };


    class OZAPI image_data_cpu : public oz::image_data_ex {
    public:
        image_data_cpu( unsigned w, unsigned h, image_format_t format, unsigned type_size = 0 );
        ~image_data_cpu();
    };


    class OZAPI image_data_gpu : public oz::image_data_ex {
    public:
        image_data_gpu( unsigned w, unsigned h, image_format_t format, unsigned type_size = 0 );
        ~image_data_gpu();
    };

}
