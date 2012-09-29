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

#include <oz/type_traits.h>

namespace oz {

    namespace detail {
        template<typename Arg, typename Result> Result convert_scalar(Arg t);
        template<> inline __host__ __device__ uchar convert_scalar( uchar t ) { return t; }
        template<> inline __host__ __device__ float convert_scalar( float t ) { return t; }
        template<> inline __host__ __device__ float convert_scalar( uchar t ) {
            return (float)t / 255.0f;
        }
        template<> inline __host__ __device__ uchar convert_scalar( float t ) {
            return (uchar)(fminf(fmaxf(255.0f * t + 0.5f, 0), 255));
        }


        template<typename Arg, typename Result, int N> struct convert_type_n;
        template<typename Arg, typename Result> struct convert_type_n<Arg,Result,1> {
            typedef typename type_traits<Arg>::scalar_type ArgS;
            typedef typename type_traits<Result>::scalar_type ResultS;
            static inline __host__ __device__ Result c( Arg s ) {
                return convert_scalar<ArgS,ResultS>(type_traits<Arg>::x(s));
            }
        };
        template<typename Arg, typename Result> struct convert_type_n<Arg,Result,2> {
            typedef typename type_traits<Arg>::scalar_type ArgS;
            typedef typename type_traits<Result>::scalar_type ResultS;
            static inline __host__ __device__ Result c( Arg s ) {
                Result q;
                q.x = convert_scalar<ArgS,ResultS>(type_traits<Arg>::x(s));
                q.y = convert_scalar<ArgS,ResultS>(type_traits<Arg>::y(s));
                return q;
            }
        };
        template<typename Arg, typename Result> struct convert_type_n<Arg,Result,3> {
            typedef typename type_traits<Arg>::scalar_type ArgS;
            typedef typename type_traits<Result>::scalar_type ResultS;
            static inline __host__ __device__ Result c( Arg s ) {
                Result q;
                q.x = convert_scalar<ArgS,ResultS>(type_traits<Arg>::x(s));
                q.y = convert_scalar<ArgS,ResultS>(type_traits<Arg>::y(s));
                q.z = convert_scalar<ArgS,ResultS>(type_traits<Arg>::z(s));
                return q;
            }
        };
        template<typename Arg, typename Result> struct convert_type_n<Arg,Result,4> {
            typedef typename type_traits<Arg>::scalar_type ArgS;
            typedef typename type_traits<Result>::scalar_type ResultS;
            static inline __host__ __device__ Result c( Arg s ) {
                Result q;
                q.x = convert_scalar<ArgS,ResultS>(type_traits<Arg>::x(s));
                q.y = convert_scalar<ArgS,ResultS>(type_traits<Arg>::y(s));
                q.z = convert_scalar<ArgS,ResultS>(type_traits<Arg>::z(s));
                q.w = convert_scalar<ArgS,ResultS>(type_traits<Arg>::w(s));
                return q;
            }
        };
    }

    template <typename Arg, typename Result>
    inline __host__ __device__ Result convert_type( Arg s ) {
        return detail::convert_type_n<Arg,Result,type_traits<Result>::N>::c(s);
    }

}
