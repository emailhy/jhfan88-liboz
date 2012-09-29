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
#include <oz/blit.h>
#include <oz/gpu_plm2.h>
#include <oz/foreach.h>
#include <oz/shuffle.h>
#include <oz/color.h>

namespace oz {

    template<typename T> struct Blit {
        gpu_plm2<T> dst_;
        uint2 od_;
        const gpu_plm2<T> src_;
        uint2 os_;

        Blit( gpu_image& dst, uint2 od, const gpu_image& src, uint2 os ) : dst_(dst), od_(od), src_(src), os_(os) {}

        inline __device__ void operator()( int ix, int iy ) {
            dst_.write(od_.x + ix, od_.y + iy, src_(os_.x + ix, os_.y + iy));
        }
    };


    void blit( gpu_image& dst, unsigned x, unsigned y, const gpu_image& src, unsigned sx, unsigned sy, unsigned sw, unsigned sh ) {
        if (dst.format() != src.format()) OZ_INVALID_FORMAT();
        if (sw > src.w()) sw = src.w();
        if (sh > src.h()) sh = src.h();
        if (sx + sw > dst.w()) sw = dst.w() - sx;
        if (sy + sh > dst.h()) sh = dst.h() - sy;
        switch (src.format()) {
            case FMT_FLOAT: {
                Blit<float> op(dst, make_uint2(x, y), src, make_uint2(sx, sy));
                foreach(sw, sh, op);
                break;
            }
            case FMT_FLOAT2: {
                Blit<float2> op(dst, make_uint2(x, y), src, make_uint2(sx, sy));
                foreach(sw, sh, op);
                break;
            }
            case FMT_FLOAT3: {
                Blit<float3> op(dst, make_uint2(x, y), src, make_uint2(sx, sy));
                foreach(sw, sh, op);
                break;
            }
            case FMT_FLOAT4: {
                Blit<float4> op(dst, make_uint2(x, y), src, make_uint2(sx, sy));
                foreach(sw, sh, op);
                break;
            }
            default:
                OZ_INVALID_FORMAT();
        }
    }


    gpu_image vstack( const gpu_image& a, const gpu_image& b, int spacing) {
        if (a.format() != b.format()) OZ_INVALID_FORMAT();
        int w = std::max(a.w(), b.w());
        int h = a.h() + b.h() + spacing;
        gpu_image dst(w, h, a.format());
        dst.clear_white();
        blit(dst, (w - a.w()) / 2, 0, a, 0, 0, a.w(), a.h());
        blit(dst, (w - b.w()) / 2, a.h() + spacing, b, 0, 0, b.w(), b.h());
        return dst;
    }


    gpu_image vstack( const gpu_image& a, const gpu_image& b, const gpu_image& c, int spacing) {
        if ((a.format() != b.format()) || (a.format() != c.format())) OZ_INVALID_FORMAT();
        int w = std::max(std::max(a.w(), b.w()), c.w());
        int h = a.h() + b.h() + c.h() + 2 * spacing;
        gpu_image dst(w, h, a.format());
        dst.clear_white();
        blit(dst, (w - a.w()) / 2, 0, a, 0, 0, a.w(), a.h());
        blit(dst, (w - b.w()) / 2, a.h() + spacing, b, 0, 0, b.w(), b.h());
        blit(dst, (w - c.w()) / 2, a.h() + b.h() + 2 * spacing, c, 0, 0, c.w(), c.h());
        return dst;
    }


    gpu_image vstack( const gpu_image& a, const gpu_image& b,
                      const gpu_image& c, const gpu_image& d, int spacing)
    {
        if ((a.format() != b.format()) || (a.format() != c.format())
            || (a.format() != d.format())) OZ_INVALID_FORMAT();

        int w = std::max(std::max(std::max(a.w(), b.w()), c.w()), d.w());
        int h = a.h() + b.h() + c.h() + d.h() + 3 * spacing;
        gpu_image dst(w, h, a.format());
        dst.clear_white();
        int y = 0;
        blit(dst, (w - a.w()) / 2, y, a, 0, 0, a.w(), a.h());
        y += a.h() + spacing;
        blit(dst, (w - b.w()) / 2, y, b, 0, 0, b.w(), b.h());
        y += b.h() + spacing;
        blit(dst, (w - c.w()) / 2, y, c, 0, 0, c.w(), c.h());
        y += c.h() + spacing;
        blit(dst, (w - c.w()) / 2, y, d, 0, 0, d.w(), d.h());
        return dst;
    }


    gpu_image vstack_alpha( const gpu_image& src, int spacing) {
        gpu_image a, b;
        switch (src.format()) {
            case FMT_FLOAT2:
                a = shuffle(src, 0);
                b = shuffle(src, 1);
                break;
            case FMT_FLOAT4:
                a = src.convert(FMT_FLOAT3);
                b = gray2rgb(shuffle(src, 3));
                break;
            default:
                OZ_INVALID_FORMAT();
        }
        return vstack(a, b, spacing);
    }


    gpu_image vstack_channel( const gpu_image& src, int spacing) {
        gpu_image a, b, c, d;
        switch (src.format()) {
            case FMT_FLOAT:
                return src;
            case FMT_FLOAT2:
                a = shuffle(src, 0);
                b = shuffle(src, 1);
                return vstack(a, b, spacing);
            case FMT_FLOAT3:
                a = shuffle(src, 0);
                b = shuffle(src, 1);
                c = shuffle(src, 2);
                return vstack(a, b, c, spacing);
            case FMT_FLOAT4:
                a = shuffle(src, 0);
                b = shuffle(src, 1);
                c = shuffle(src, 2);
                d = shuffle(src, 3);
                return vstack(a, b, c, d, spacing);
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
