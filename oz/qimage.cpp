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
#include <oz/qimage.h>
#include <qimage.h>


QImage oz::to_qimage( const cpu_image& src ) {
    cpu_image s = src;
    switch (s.format()) {
        case FMT_FLOAT:
            s = s.convert(FMT_UCHAR);
            break;
        case FMT_FLOAT3:
            s = s.convert(FMT_UCHAR4);
            break;
        case FMT_FLOAT4:
            s = s.convert(FMT_UCHAR4);
            break;
        default:
            break;
    }

    QImage q;
    switch (s.format()) {
        case FMT_UCHAR:
            q = QImage(s.w(), s.h(), QImage::Format_Indexed8);
            for (unsigned i = 0; i < 256; ++i) q.setColor(i, 0xff000000 | (i << 16) | (i << 8) | i);
            s.get((uchar*)q.bits(), q.bytesPerLine());
            break;
        case FMT_UCHAR3:
            q = QImage(s.w(), s.h(), QImage::Format_RGB32);
            s.get((uchar4*)q.bits(), q.bytesPerLine());
            break;
        case FMT_UCHAR4:
            q = QImage(s.w(), s.h(), QImage::Format_ARGB32);
            s.get((uchar4*)q.bits(), q.bytesPerLine());
            break;
        default:
            break;
    }

    return q;
}


oz::cpu_image oz::from_qimage( const QImage& src, image_format_t format ) {
    cpu_image dst;
    QImage q = src;
    if ((src.format() != QImage::Format_Indexed8) &&
        (src.format() != QImage::Format_RGB32) &&
        (src.format() != QImage::Format_ARGB32))
    {
        q = q.convertToFormat(QImage::Format_ARGB32);
    }

    switch (src.format()){
        case QImage::Format_Indexed8:
            dst = cpu_image((uchar*)src.bits(), src.bytesPerLine(), src.width(), src.height());
            break;
        case QImage::Format_RGB32:
            dst = cpu_image((uchar4*)src.bits(), src.bytesPerLine(), src.width(), src.height(), true);
            break;
        case QImage::Format_ARGB32:
            dst = cpu_image((uchar4*)src.bits(), src.bytesPerLine(), src.width(), src.height(), true);
            break;
        default:
            assert(0);
    }

    if (format != FMT_INVALID) dst = dst.convert(format);
    return dst;
}
