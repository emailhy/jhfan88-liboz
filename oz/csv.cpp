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
#include <oz/csv.h>
#include <iostream>
#include <fstream>

namespace oz {

    void csv_write( const std::string& header, const std::vector<double>& src, const std::string& path ) {
        std::ofstream f;
        f.open(path.c_str());
        if (!header.empty()) {
            f << header << "\n";
        }
        for (std::vector<double>::const_iterator i = src.begin(); i != src.end(); ++i) {
            f << *i << "\n";
        }
        f.close();
    }


    void csv_write( const std::string& hx, const std::string& hy, const std::vector<double2>& src, const std::string& path ) {
        std::ofstream f;
        f.open(path.c_str());
        if (!hx.empty() || !hy.empty()) {
            f << hx << ", " << hy << "\n";
        }
        for (std::vector<double2>::const_iterator i = src.begin(); i != src.end(); ++i) {
            f << i->x << ", " << i->y << "\n";
        }
        f.close();
    }


    void csv_write( const std::vector<std::string>& header, const cpu_image& src, const std::string& path ) {
        if (src.format() != FMT_FLOAT) OZ_INVALID_FORMAT();

        std::ofstream f;
        f.open(path.c_str());

        if (header.size()) {
            if (header.size() != src.w()) OZ_X() << "Invalid header size!";
            for (unsigned i = 0; i < header.size(); ++i) {
                f << header[i];
                if (i != header.size() - 1) {
                    f << ", ";
                }
            }
            f << "\n";
        }

        for (unsigned j = 0; j < src.h(); ++j) {
            for (unsigned i = 0; i < src.w(); ++i) {
                f << src.at<float>(i,j);
                if (i != src.w() - 1) {
                    f << ", ";
                }
            }
            f << "\n";
        }

        f.close();
    }


    void csv_write( const cpu_image& src, const std::string& path ) {
        std::vector<std::string> header;
        csv_write(header, src, path);
    }

}
