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

#include <oz/config.h>
#include <exception>
#include <string>
#include <cuda_runtime.h>

namespace oz {

    class exception : public std::exception {
    public:
        OZAPI exception();
        OZAPI exception( const char* function, const char* file, int line );
        OZAPI exception( const exception& e );
        OZAPI virtual ~exception() throw();

        OZAPI virtual const char* what() const throw();

        OZAPI exception& operator<<( signed char value );
        OZAPI exception& operator<<( unsigned char value );
        OZAPI exception& operator<<( signed short value );
        OZAPI exception& operator<<( unsigned short value );
        OZAPI exception& operator<<( signed int value );
        OZAPI exception& operator<<( unsigned int value );
        OZAPI exception& operator<<( signed long long value );
        OZAPI exception& operator<<( unsigned long long value );
        OZAPI exception& operator<<( float value );
        OZAPI exception& operator<<( double value );
        OZAPI exception& operator<<( const char* value );
        OZAPI exception& operator<<( const std::string& str );
        OZAPI exception& operator<<( cudaError_t err );

    private:
        struct Stream;
        Stream *stream_;
    };

}

#define OZ_X() throw ::oz::exception(__FUNCTION__, __FILE__, __LINE__)

#define OZ_CHECK_FORMAT( have_, want_ ) \
    if ((have_) != (want_)) \
        OZ_X() << image_format_invalid_msg(have_, want_)

#define OZ_CHECK_FORMAT2( have_, want0_, want1_ ) \
    if (((have_) != (want0_)) && ((have_) != (want1_))) \
        OZ_X() << image_format_invalid_msg(have_, want0_, want1_)

#define OZ_INVALID_FORMAT() OZ_X() << "Invalid format!"
#define OZ_INVALID_SIZE() OZ_X() << "Invalid size!"

#define OZ_CUDA_ERROR_CHECK() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { OZ_X() << err; }}

#define OZ_CUDA_SAFE_CALL( call ) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { OZ_X() << err; }}
