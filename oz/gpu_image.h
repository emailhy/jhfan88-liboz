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

namespace oz {

    class cpu_image;

    class OZAPI gpu_image {
    public:
        gpu_image() : d_(0) { }
        gpu_image( unsigned w, unsigned h, image_format_t format, unsigned type_size = 0 );
        gpu_image( NppiSize size, image_format_t format, unsigned type_size = 0 );

        gpu_image( unsigned w, unsigned h, float value );
        gpu_image( unsigned w, unsigned h, float2 value );
        gpu_image( unsigned w, unsigned h, float3 value );
        gpu_image( unsigned w, unsigned h, float4 value );

        gpu_image( const gpu_image& img ) {
            if (img.d_) img.d_->add_ref();
            d_ = img.d_;
        }

        gpu_image( const cpu_image& img );

        gpu_image( image_data_gpu *d ) {
            if (d) d->add_ref();
            d_ = d;
        }

        ~gpu_image() {
            if (d_) d_->release();
            d_ = 0;
        }

        const gpu_image& operator=( const gpu_image& img ) {
            if (img.d_) img.d_->add_ref();
            if (d_) d_->release();
            d_ = img.d_;
            return *this;
        }

        const gpu_image& operator=( const cpu_image& img );

        void swap( gpu_image& img ) {
            image_data_gpu *tmp = d_;
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
        unsigned stride() const { return d_->stride(); }
        unsigned w() const { return d_->w(); }
        unsigned h() const { return d_->h(); }
        unsigned N() const { return d_->N(); }
        NppiSize size() const { return d_->size(); }

        void clear();
        void clear_white();
        void fill( float value, int x, int y, int w, int h );
        void fill( float2 value, int x, int y, int w, int h );
        void fill( float3 value, int x, int y, int w, int h );
        void fill( float4 value, int x, int y, int w, int h );

        gpu_image clone() const;
        cpu_image cpu() const;
        gpu_image convert( image_format_t format, bool clone=false ) const;

    private:
        image_data_gpu *d_;
    };

    OZAPI gpu_image operator-( const gpu_image& src );
    OZAPI gpu_image operator+( const gpu_image& a, const gpu_image& b );
    OZAPI gpu_image operator+( const gpu_image& a, float b );
    OZAPI gpu_image operator-( const gpu_image& a, const gpu_image& b );
    OZAPI gpu_image operator-( const gpu_image& a, float b );
    OZAPI gpu_image operator*( const gpu_image& a, const gpu_image& b );
    OZAPI gpu_image operator*( const gpu_image& src, float k );
    OZAPI gpu_image operator*( float k, const gpu_image& src );
    OZAPI gpu_image operator*( const gpu_image& src, float2 k );
    OZAPI gpu_image operator*( float2 k, const gpu_image& src );
    OZAPI gpu_image operator*( const gpu_image& src, float3 k );
    OZAPI gpu_image operator*( float3 k, const gpu_image& src );
    OZAPI gpu_image operator*( const gpu_image& src, float4 k );
    OZAPI gpu_image operator*( float4 k, const gpu_image& src );
    OZAPI gpu_image operator/( const gpu_image& a, const gpu_image& b );
    OZAPI gpu_image operator/( const gpu_image& src, float k );

    OZAPI gpu_image adjust( const gpu_image& src, float a, float b );
    OZAPI gpu_image invert( const gpu_image& src );
    OZAPI gpu_image saturate( const gpu_image& src );
    OZAPI gpu_image clamp( const gpu_image& src, float a, float b );
    OZAPI gpu_image clamp( const gpu_image& src, float2 a, float2 b );
    OZAPI gpu_image clamp( const gpu_image& src, float3 a, float3 b );
    OZAPI gpu_image clamp( const gpu_image& src, float4 a, float4 b );
    OZAPI gpu_image lerp( const gpu_image& src0,const gpu_image& src1, float t );
    OZAPI gpu_image abs( const gpu_image& src );
    OZAPI gpu_image abs2( const gpu_image& src );
    OZAPI gpu_image sqrt( const gpu_image& src );
    OZAPI gpu_image sqr( const gpu_image& src );
    OZAPI gpu_image pow( const gpu_image& src, float y );
    OZAPI gpu_image log( const gpu_image& src );
    OZAPI gpu_image abs_diff( const gpu_image& src0, const gpu_image& src1 );
    OZAPI gpu_image log_abs( const gpu_image& src );
    OZAPI gpu_image circshift( const gpu_image& src, int dx, int dy );

}
