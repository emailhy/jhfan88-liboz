//
// Copyright (c) 1995-2011 by Jan Eric Kyprianidis <www.kyprianidis.com>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
#include <oz/io.h>
#include <cstdio>
#include <cstring>
#include <memory>


/*--Descriptor flags---------------*/
#define ds_BottomLeft  0x00
#define ds_BottomRight 0x01
#define ds_TopLeft     0x02
#define ds_TopRight    0x03

/*--Color map types----------------*/
#define cm_NoColorMap  0
#define cm_ColorMap    1

/*--Image data types---------------*/
#define im_NoImageData    0
#define im_ColMapImage    1
#define im_RgbImage       2
#define im_MonoImage      3
#define im_ColMapImageRLE 9
#define im_RgbImageRLE    10
#define im_MonoImageRLE   11


static unsigned short fgetword(FILE *f) {
    unsigned char b[2];
    fread(b, 2, 1, f);
    return ((unsigned short)b[1] << 8) | (unsigned short)b[0];
}


static void  fputword(unsigned short s, FILE *f) {
    unsigned char b[2];
    b[0] = (unsigned char)(s & 0xff);
    b[1] = (unsigned char)((s >> 8) & 0xff);
    fwrite(b, 2, 1, f);
}


oz::cpu_image tga_read( const std::string& path ) {
    FILE *f = 0;
    try {
        int idLen;
        int colorMapType;
        int imageType;
        int firstColor;
        int colorMapLen;
        int colorMapBits;
        int xOrgin;
        int yOrgin;
        int iw;
        int ih;
        int bitsPerPix;
        int descriptor;
        int bb;

        if ((f = fopen(path.c_str(), "rb")) == NULL)
            OZ_X() << "File not found!";

        idLen = fgetc(f);
        colorMapType = fgetc(f);
        imageType = fgetc(f);
        firstColor = fgetword(f);
        colorMapLen = fgetword(f);
        colorMapBits = fgetc(f);
        xOrgin = fgetword(f);
        yOrgin = fgetword(f);
        iw = fgetword(f);
        ih = fgetword(f);
        bitsPerPix = fgetc(f);
        descriptor = fgetc(f);

        switch(imageType) {
            case im_MonoImage:
            case im_ColMapImage:
            case im_RgbImage:
            case im_MonoImageRLE:
            case im_ColMapImageRLE:
            case im_RgbImageRLE:
                break;
            default:
                OZ_X() << "Invalid image format!";
        }
        switch (bitsPerPix) {
            case 8:
            case 24:
            case 32:
                bb = bitsPerPix / 8;
                break;
            default:
                OZ_X() << "Unsupported bits per pixel!";
        }

        if (idLen)
            fseek(f, idLen, SEEK_CUR);

        unsigned char colorMap[256][3];
        if (colorMapType == cm_ColorMap) {
            memset(colorMap, 0, sizeof(colorMap));
            if ((colorMapBits != 24) || (colorMapLen + firstColor > 256)) {
                OZ_X() << "Unsupported format!";
            }
            unsigned char* pp = colorMap[firstColor];
            for (int i = 0; i < colorMapLen; i++) {
                *pp++ = fgetc(f);
                *pp++ = fgetc(f);
                *pp++ = fgetc(f);
            }
        }

        std::auto_ptr<uchar> data(new uchar[iw * bb * ih]);

        if ((imageType == im_MonoImage) || (imageType == im_ColMapImage) || (imageType == im_RgbImage)) {
            fread(data.get(), ih, iw*bb, f);
        } else {
            unsigned char* p = data.get();
            int n = 0;
            int rle = 0;
            int cp[4];
            int c;
            for (int j = 0; j < ih; ++j) {
                for (int i = 0; i < iw; ++i, p += bb) {
                    if (n == 0) {
                        c = fgetc(f);
                        n = (c & 0x7F) + 1;
                        if (c&0x80) {
                            for (int k = 0; k < bb; k++) cp[k] = fgetc(f);
                            rle = 1;
                        } else {
                            rle = 0;
                        }
                    }
                    if (rle)
                        for (int k = 0; k < bb; k++) p[k] = cp[k];
                    else
                        for (int k = 0; k < bb; k++) p[k] = fgetc(f);
                    n--;
                }
            }
        }
        if (ferror(f)) {
            OZ_X() << "IO error!";
        }
        fclose(f);
        f = 0;

        int ix, iy;
        unsigned char *q;
        switch ((descriptor >> 4) & 0x3) {
            case ds_BottomLeft:
                ix = bb;
                iy = -2 * iw * bb;
                q = data.get() + iw * bb * (ih - 1);
                break;
            case ds_BottomRight:
                ix = -bb;
                iy = 0;
                q = data.get() + (iw - 1) * bb * (ih);
                break;
            case ds_TopLeft:
                ix = bb;
                iy = 0;
                q = data.get();
                break;
            case ds_TopRight:
                ix = -bb;
                iy = 2 * iw * bb;
                q = data.get() + (iw - 1) * bb;
                break;
            default:
                OZ_X() << "Corrupt file data!";
        }

        oz::cpu_image dst;
        switch (bitsPerPix) {
            case 8: {
                if (colorMapType == cm_ColorMap) {
                    dst = oz::cpu_image(iw, ih, oz::FMT_UCHAR3);
                    uchar* p = dst.ptr<uchar>();
                    for (int j = 0; j < ih; ++j) {
                        for (int i = 0; i < iw; ++i) {
                            for (int k = 0; k < 3; ++k) {
                                p[k] = colorMap[*q][k];
                            }
                            p[3] = 0xff;
                            p += 4;
                            q += ix;
                        }
                        p += dst.padding();
                        q += iy;
                    }
                } else {
                    dst = oz::cpu_image(iw, ih, oz::FMT_UCHAR);
                    uchar* p = dst.ptr<uchar>();
                    for (int j = 0; j < ih; ++j) {
                        for (int i = 0; i < iw; ++i) {
                            *p = *q;
                            p += 1;
                            q += ix;
                        }
                        p += dst.padding();
                        q += iy;
                    }
                }
                break;
            }

            case 24: {
                dst = oz::cpu_image(iw, ih, oz::FMT_UCHAR3);
                uchar* p = dst.ptr<uchar>();
                for (int j = 0; j < ih; ++j) {
                    for (int i = 0; i < iw; ++i) {
                        p[0] = q[0];
                        p[1] = q[1];
                        p[2] = q[2];
                        p[3] = 0xff;
                        p += 4;
                        q += ix;
                    }
                    p += dst.padding();
                    q += iy;
                }
                break;
            }

            case 32: {
                dst = oz::cpu_image(iw, ih, oz::FMT_UCHAR4);
                uchar* p = dst.ptr<uchar>();
                for (int j = 0; j < ih; ++j) {
                    for (int i = 0; i < iw; ++i) {
                        p[0] = q[0];
                        p[1] = q[1];
                        p[2] = q[2];
                        p[3] = q[3];
                        p += 4;
                        q += ix;
                    }
                    p += dst.padding();
                    q += iy;
                }
                break;
            }
        }

        return dst;
    }
    catch (std::exception&) {
        if (f) fclose(f);
        throw;
    }
}


void tga_write( const oz::cpu_image& src, const std::string& path ) {
    FILE *f = 0;
    try {
        int colorMapType = 0;
        int imageType;
        int colorMapLen = 0;
        int colorMapBits = 0;

        int bitsPerPix;
        int bb;
        switch (src.format()) {
            case oz::FMT_UCHAR:
                imageType = im_MonoImage;
                bitsPerPix = 8;
                bb = 1;
                break;
            case oz::FMT_UCHAR3:
                imageType = im_RgbImage;
                bitsPerPix = 24;
                bb = 3;
                break;
            case oz::FMT_UCHAR4:
                imageType = im_RgbImage;
                bitsPerPix = 32;
                bb = 4;
                break;
            default:
                OZ_X() << "Unsupported format!";
        }

        if ((f = fopen(path.c_str(), "wb+")) == NULL)
            OZ_X() << "IO error!";

        fputc(0, f);
        fputc(colorMapType, f);
        fputc(imageType, f);
        fputword(0, f);
        fputword(colorMapLen, f);
        fputc(colorMapBits, f);
        fputword(0, f);
        fputword(0, f);
        fputword(src.w(), f);
        fputword(src.h(), f);
        fputc(bitsPerPix, f);
        fputc(ds_TopLeft << 4, f);

        if (src.format() != oz::FMT_UCHAR3) {
            uchar *p = src.ptr<uchar>();
            for (unsigned j = 0; j < src.h(); ++j) {
                if (fwrite(p, src.row_size(), 1, f) != 1) {
                    OZ_X() << "IO error!";
                }
                p += src.pitch();
            }
        } else {
            std::auto_ptr<uchar3> buf( new uchar3[src.N()]);
            src.get(buf.get(), sizeof(uchar3)*src.w());
            if (fwrite(buf.get(),sizeof(uchar3)*src.N(), 1, f) != 1) {
                OZ_X() << "IO error!";
            }
        }

        fclose(f);
    }
    catch(std::exception&) {
        if (f) fclose(f);
        throw;
    }
}
