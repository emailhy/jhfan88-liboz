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
#include <oz/exception.h>
#include <oz/log.h>
#include <sstream>


struct oz::exception::Stream {
    Stream() { ref_count = 1; }
    int ref_count;
    std::stringstream s;
    std::string tmp;
};


oz::exception::exception() {
    stream_ = new Stream;
}


oz::exception::exception( const exception& e ) {
    stream_ = e.stream_;
    ++stream_->ref_count;
}


oz::exception::~exception() throw() {
    --stream_->ref_count;
    if (stream_->ref_count == 0) {
        delete stream_;
    }
    stream_ = 0;
}


oz::exception::exception( const char* function, const char* file, int line ) {
    log_stack();
    stream_ = new Stream;
    stream_->s << file << "(" << line << "): ";
    if (function) {
        stream_->s << function << " -- ";
    }
}


const char* oz::exception::what() const throw() {
    stream_->tmp = stream_->s.str();
    return stream_->tmp.c_str();
}


oz::exception& oz::exception::operator<<( signed char value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( unsigned char value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( signed short value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( unsigned short value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( signed int value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( unsigned int value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( signed long long value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( unsigned long long value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( float value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( double value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( const char *value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( const std::string& value ) {
    stream_->s << value;
    return *this;
}


oz::exception& oz::exception::operator<<( cudaError_t err ) {
    stream_->s << "CUDA error (" << cudaGetErrorString(err) << ")";
    return *this;
}
