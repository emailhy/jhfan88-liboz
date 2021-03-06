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
#include <oz/io.h>
#include <tiffio.h>


static const unsigned char icc_profile[520] = {
    0x00, 0x00, 0x02, 0x08, 0x41, 0x44, 0x42, 0x45, 0x02, 0x10, 0x00, 0x00, 0x6d, 0x6e, 0x74, 0x72,
    0x52, 0x47, 0x42, 0x20, 0x58, 0x59, 0x5a, 0x20, 0x07, 0xdb, 0x00, 0x05, 0x00, 0x1e, 0x00, 0x07,
    0x00, 0x2c, 0x00, 0x38, 0x61, 0x63, 0x73, 0x70, 0x41, 0x50, 0x50, 0x4c, 0x00, 0x00, 0x00, 0x00,
    0x6e, 0x6f, 0x6e, 0x65, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf6, 0xd6, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xd3, 0x2c,
    0x41, 0x44, 0x42, 0x45, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x09, 0x63, 0x70, 0x72, 0x74, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00, 0x32,
    0x64, 0x65, 0x73, 0x63, 0x00, 0x00, 0x01, 0x24, 0x00, 0x00, 0x00, 0x81, 0x77, 0x74, 0x70, 0x74,
    0x00, 0x00, 0x01, 0xa8, 0x00, 0x00, 0x00, 0x14, 0x72, 0x58, 0x59, 0x5a, 0x00, 0x00, 0x01, 0xbc,
    0x00, 0x00, 0x00, 0x14, 0x67, 0x58, 0x59, 0x5a, 0x00, 0x00, 0x01, 0xd0, 0x00, 0x00, 0x00, 0x14,
    0x62, 0x58, 0x59, 0x5a, 0x00, 0x00, 0x01, 0xe4, 0x00, 0x00, 0x00, 0x14, 0x72, 0x54, 0x52, 0x43,
    0x00, 0x00, 0x01, 0xf8, 0x00, 0x00, 0x00, 0x0e, 0x67, 0x54, 0x52, 0x43, 0x00, 0x00, 0x01, 0xf8,
    0x00, 0x00, 0x00, 0x0e, 0x62, 0x54, 0x52, 0x43, 0x00, 0x00, 0x01, 0xf8, 0x00, 0x00, 0x00, 0x0e,
    0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, 0x00, 0x43, 0x6f, 0x70, 0x79, 0x72, 0x69, 0x67, 0x68,
    0x74, 0x20, 0x32, 0x30, 0x31, 0x31, 0x20, 0x41, 0x64, 0x6f, 0x62, 0x65, 0x20, 0x53, 0x79, 0x73,
    0x74, 0x65, 0x6d, 0x73, 0x20, 0x49, 0x6e, 0x63, 0x6f, 0x72, 0x70, 0x6f, 0x72, 0x61, 0x74, 0x65,
    0x64, 0x00, 0x00, 0x00, 0x64, 0x65, 0x73, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x27,
    0x73, 0x52, 0x47, 0x42, 0x20, 0x49, 0x45, 0x43, 0x36, 0x31, 0x39, 0x36, 0x36, 0x2d, 0x32, 0x2e,
    0x31, 0x20, 0x28, 0x4c, 0x69, 0x6e, 0x65, 0x61, 0x72, 0x20, 0x52, 0x47, 0x42, 0x20, 0x50, 0x72,
    0x6f, 0x66, 0x69, 0x6c, 0x65, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x58, 0x59, 0x5a, 0x20, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0xf3, 0x52, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x16, 0xcc, 0x58, 0x59, 0x5a, 0x20,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6f, 0xa0, 0x00, 0x00, 0x38, 0xf5, 0x00, 0x00, 0x03, 0x90,
    0x58, 0x59, 0x5a, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x62, 0x97, 0x00, 0x00, 0xb7, 0x87,
    0x00, 0x00, 0x18, 0xd8, 0x58, 0x59, 0x5a, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x9f,
    0x00, 0x00, 0x0f, 0x84, 0x00, 0x00, 0xb6, 0xc4, 0x63, 0x75, 0x72, 0x76, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00
};


oz::cpu_image tiff_read( const std::string& path ) {
    TIFF *in = 0;
    try {
        in = TIFFOpen(path.c_str(), "r");
        if (!in) OZ_X() << "Opening file '" << path << "' failed!";

        int w,h;
        if (TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &w) != 1) OZ_X() << "Invalid format!";
        if (TIFFGetField(in, TIFFTAG_IMAGELENGTH,&h) != 1) OZ_X() << "Invalid format!";

        uint16 bps, photometric, spp;
        if (TIFFGetField(in, TIFFTAG_BITSPERSAMPLE, &bps) !=1) OZ_X() << "Unsupported format!";
        if (TIFFGetField(in, TIFFTAG_PHOTOMETRIC, &photometric) !=1)  OZ_X() << "Unsupported format!";
        if (TIFFGetField(in, TIFFTAG_SAMPLESPERPIXEL, &spp) !=1)  OZ_X() << "Unsupported format!";

        /*
        uint32 profile_size;
        unsigned char *profile_data;
        if (TIFFGetField(in, TIFFTAG_ICCPROFILE, &profile_size, &profile_data)) {
            FILE *f = fopen("C:/Users/jkyprian/Desktop/profile.h", "wt+");
            fprintf(f, "%d\n", profile_size);
            for (unsigned i = 0; i < profile_size; ++i) {
                if (i % 16 == 0) fprintf(f, "\n");
                fprintf(f, "0x%02x, ", profile_data[i]);
            }
            fclose(f);
        }
        */

        //TIFFGetField(in, TIFFTAG_EXTRASAMPLES, &xs, &xs_ptr);
        //xs=0;
        //uint16* xs_ptr=0;
        //if (xs != 1) break;

        if ((bps != 32) || (photometric != PHOTOMETRIC_RGB) || (spp != 4))
            OZ_X() << "Unsupported format!";

        //tmsize_t scanlineSize = TIFFScanlineSize(in);
        oz::cpu_image dst(w, h, oz::FMT_FLOAT4);
        int j;
        for (j = 0; j < h; ++j) {
            if (TIFFReadScanline(in, dst.scan_line<float4>(j), j) !=1 ) break;
        }
        if (j != h) OZ_X() << "Reading scan line failed!";

        return dst;
    }
    catch (std::exception&) {
        if (in) TIFFClose(in);
        throw;
    }
}


void tiff_write( const oz::cpu_image& src, const std::string& path ) {
    if ((src.format() != oz::FMT_FLOAT4) ||
        (src.row_size() != src.pitch())) OZ_INVALID_FORMAT();

    TIFF *out = 0;
    try {
        out = TIFFOpen(path.c_str(), "w");
        if (!out) OZ_X() << "Opening file '" << path << "' failed!";

        int w = src.w();
        int h = src.h();
        uint16 bps = 32;
        uint16 photometric = PHOTOMETRIC_RGB;
        uint16 spp = 4;
        uint16 xs = 1;
        uint16 xs_ptr[1] = { 0 };
        uint16 pc = PLANARCONFIG_CONTIG;
        uint32 rowsperstrip = src.h();
        uint32 profile_size = sizeof(icc_profile);

        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, w);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, h);
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, spp);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, bps);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, pc);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, photometric);
        //TIFFSetField(out, TIFFTAG_RESOLUTIONUNIT, RESUNIT_NONE);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, h);
        //TIFFSetField(out, TIFFTAG_EXTRASAMPLES, xs, xs_ptr);
        TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        TIFFSetField(out, TIFFTAG_ICCPROFILE, profile_size, icc_profile);

        TIFFWriteEncodedStrip(out, 0, (void*)src.ptr(), 16* src.w() * src.h());

        TIFFClose(out);
    }
    catch (std::exception&) {
        if (out) TIFFClose(out);
        throw;
    }
}
