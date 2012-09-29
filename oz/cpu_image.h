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

#include <oz/image_data.h>
#include <oz/type_traits.h>

namespace oz {

    class gpu_image;

    class OZAPI cpu_image {
    public:
        cpu_image() : d_(0) { }
        cpu_image( unsigned w, unsigned h, image_format_t format, unsigned type_size = 0 );
        cpu_image( NppiSize size, image_format_t format, unsigned type_size = 0 );

        cpu_image( const uchar *src, unsigned src_pitch, unsigned w, unsigned h );
        cpu_image( const uchar2 *src, unsigned src_pitch, unsigned w, unsigned h );
        cpu_image( const uchar3 *src, unsigned src_pitch, unsigned w, unsigned h );
        cpu_image( const uchar4 *src, unsigned src_pitch, unsigned w, unsigned h, bool ignore_alpha );
        cpu_image( const float *src, unsigned src_pitch, unsigned w, unsigned h );
        cpu_image( const float2 *src, unsigned src_pitch, unsigned w, unsigned h );
        cpu_image( const float3 *src, unsigned src_pitch, unsigned w, unsigned h );
        cpu_image( const float4 *src, unsigned src_pitch, unsigned w, unsigned h, bool ignore_alpha );

        cpu_image( const cpu_image& img ) {
            if (img.d_) img.d_->add_ref();
            d_ = img.d_;
        }

        cpu_image( const gpu_image& img );

        explicit cpu_image( image_data_cpu *d ) {
            if (d) d->add_ref();
            d_ = d;
        }

        ~cpu_image() {
            if (d_) d_->release();
            d_ = 0;
        }

        const cpu_image& operator=(const cpu_image& img) {
            if (img.d_) img.d_->add_ref();
            if (d_) d_->release();
            d_ = img.d_;
            return *this;
        }

        const cpu_image& operator=( const gpu_image& img );

        void swap( cpu_image& img ) {
            image_data_cpu *tmp = d_;
            d_ = img.d_;
            img.d_ = tmp;
        }

        bool is_valid() const { return d_ != 0; }
        template<typename T> T* ptr() const { return static_cast<T*>(d_->ptr()); }
        void* ptr() const { return d_->ptr(); }
        image_format_t format() const { return d_? d_->format() : FMT_INVALID; }
        unsigned type_size() const { return d_->type_size(); }
        unsigned pitch() const { return d_->pitch(); }
        unsigned row_size() const { return d_->row_size(); }
        unsigned padding() const { return d_->padding(); }
        unsigned w() const { return d_->w(); }
        unsigned h() const { return d_->h(); }
        unsigned N() const { return d_->N(); }
        NppiSize size() const { return d_->size(); }

        void clear();

        cpu_image clone() const;
        gpu_image gpu() const;
        cpu_image convert( image_format_t format, bool clone=false ) const;
        cpu_image copy( int x1, int y1, int x2, int y2 ) const;

        void get( uchar *dst, unsigned dst_pitch ) const;
        void get( uchar2 *dst, unsigned dst_pitch ) const;
        void get( uchar3 *dst, unsigned dst_pitch ) const;
        void get( uchar4 *dst, unsigned dst_pitch ) const;
        void get( float *dst, unsigned dst_pitch ) const;
        void get( float2 *dst, unsigned dst_pitch ) const;
        void get( float3 *dst, unsigned dst_pitch ) const;
        void get( float4 *dst, unsigned dst_pitch ) const;

        template<typename T> T* scan_line( int y ) {
            return reinterpret_cast<T*>(this->ptr<char>() + y * this->pitch());
        }

        template<typename T> const T* scan_line( int y ) const {
            return reinterpret_cast<const T*>(this->ptr<char>() + y * this->pitch());
        }

        template<typename T> T& at(int x, int y) {
            typedef typename type_traits<T>::texture_type TT;
            if (x < 0) x = 0; else if (x >= (int)this->w()) x = this->w() - 1;
            if (y < 0) y = 0; else if (y >= (int)this->h()) y = this->h() - 1;
            return *reinterpret_cast<T*>(&this->scan_line<TT>(y)[x]);
        }

        template<typename T> const T& at(int x, int y) const {
            typedef typename type_traits<T>::texture_type TT;
            if (x < 0) x = 0; else if (x >= (int)this->w()) x = this->w() - 1;
            if (y < 0) y = 0; else if (y >= (int)this->h()) y = this->h() - 1;
            return *reinterpret_cast<const T*>(&this->scan_line<TT>(y)[x]);
        }

        template<typename T> const T& sample_nearest(float x, float y) const {
            return this->at<T>((int)x, (int)y);
        }

        // TODO: optimize
        template<typename T> T sample_linear(float x, float y) const {
            x -= 0.5f;
            y -= 0.5f;

            int x0 = (int)x;
            int y0 = (int)y;
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float fx = x - floor(x);
            float fy = y - floor(y);

            T c0 = this->at<T>(x0, y0);
            T c1 = this->at<T>(x1, y0);
            T c2 = this->at<T>(x0, y1);
            T c3 = this->at<T>(x1, y1);

            return (1 - fy) * ((1 - fx) * c0 + fx * c1) + fy * ((1 - fx) * c2 + fx * c3);
        }

    private:
        image_data_cpu *d_;
    };

}
