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
#include <algorithm>
#include <QtGui/qimage.h>
#include <oz/qimage.h>
#include <cstdio>


oz::cpu_image tga_read( const std::string& path );
void tga_write( const oz::cpu_image& src, const std::string& path );


static std::string file_extension( const std::string& path ) {
    std::string ext;
    if (path.length() > 4) {
        ext = path.substr(path.length() - 4, 4);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }
    return ext;
}


oz::cpu_image oz::imread( const std::string& path ) {
    std::string ext = file_extension(path);
    if (ext == ".tga") {
        cpu_image I = tga_read(path);
        return I;
    } else if ((ext == ".png") || (ext == ".jpg") || (ext == ".jpeg")) {
        QImage img(path.c_str());
        return from_qimage(img);
    } else {
        OZ_X() << "Unsupported image format: '" << path << "'";
    }
}


oz::cpu_image oz::imread( const char* path, ... ) {
   va_list args;
   char buffer[256];
   va_start(args, path);
#ifdef _MSC_VER
   vsprintf_s(buffer, 256, path, args);
#else
   vsnprintf(buffer, 256, path, args);
#endif
   return imread(std::string(buffer));
}


void oz::imwrite( const cpu_image& src, const std::string& path ) {
    std::string ext = file_extension(path);
    if (ext == ".tga") {
        tga_write( src, path );
    } else if ((ext == ".png") || (ext == ".jpg") || (ext == ".jpeg")) {
        QImage img = to_qimage(src);
        if (!img.save(QString::fromStdString(path))) {
            OZ_X() << "Saving '" << path << "' failed!";
        }
    } else {
        OZ_X() << "Unsupported image format: '" << path << "'";
    }
}


void oz::imwrite( const cpu_image& src, const char* path, ... ) {
   va_list args;
   char buffer[256];
   va_start(args, path);
#ifdef _MSC_VER
   vsprintf_s(buffer, 256, path, args);
#else
   vsnprintf(buffer, 256, path, args);
#endif
   imwrite(src, std::string(buffer));
}
