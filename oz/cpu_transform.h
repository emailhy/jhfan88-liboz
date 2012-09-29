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

#include <oz/transform.h>
#include <oz/cpuimage.h>

namespace oz {

    template<typename Arg, typename Result, typename F>
    void transform_unary( const cpu_image& src, cpu_image& dst, const F& f ) {
        if ((src.format() != type_traits<Arg>::format()) ||
            (dst.format() != type_traits<Result>::format())) OZ_INVALID_FORMAT();
        char *pp = dst.ptr<char>();
        char *qq = src.ptr<char>();
        for (unsigned j = 0; j < dst.h(); ++j) {
            typename type_traits<Arg>::texture_type *q = (typename type_traits<Arg>::texture_type*)qq;
            typename type_traits<Result>::texture_type *p = (typename type_traits<Result>::texture_type*)pp;
            for (unsigned i = 0; i < dst.w(); ++i) {
                *reinterpret_cast<Result*>(p) = f(*reinterpret_cast<Arg*>(q));
                ++p;
                ++q;
            }
            pp += dst.pitch();
            qq += src.pitch();
        }
    }

    template<typename Arg, typename Result, typename F>
    cpu_image transform_unary( const cpu_image& src, const F& f ) {
        cpu_image dst(src.size(), type_traits<Result>::format());
        transform_inplace<unary_function>(src, dst, f);
        return dst;
    }

    template<typename F>
    cpu_image transform( const cpu_image& src, const F& f ) {
        typedef typename F::argument_type Arg;
        typedef typename F::result_type Result;
        return transform_unary<Arg,Result,F>(src, f);
    }

}
