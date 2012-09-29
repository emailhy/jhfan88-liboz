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
#include <oz/matfile.h>
#include "mat.h"
#include <algorithm>


void oz::matfile_write( const std::map<std::string, cpu_image> vars, const std::string& path ) {
    MATFile *pmat = 0;
    mxArray *pa = 0;
    try {
        pmat = matOpen(path.c_str(), "w");
        if (!pmat)
            OZ_X() << "Creating file '" << path << "' failed!";

        for (std::map<std::string, cpu_image>::const_iterator i = vars.begin(); i != vars.end(); ++i) {
            std::string name = i->first;
            std::replace(name.begin(), name.end(), '-', '_');

            const oz::cpu_image& img = i->second;
            if (img.format() != FMT_FLOAT) OZ_X() << "Invalid format!";

            pa = mxCreateNumericMatrix(img.w(), img.h(), mxSINGLE_CLASS, mxREAL);
            if (!pa)
                OZ_X() << "Creation of matrix for '" << name << "'failed!";

            float *p = (float*)mxGetData(pa);
            for (unsigned i = 0; i < img.w(); ++i) {
                float *q = img.ptr<float>() + i;
                for (unsigned j = 0; j < img.h(); ++j) {
                    *p++ = *q;
                    q += img.pitch() / sizeof(float);
                }
            }

            if (matPutVariable(pmat, name.c_str(), pa) != 0)
                OZ_X() << "Putting variable '" << name << "'failed!";

            mxDestroyArray(pa);
        }

        matClose(pmat);
    }
    catch (std::exception&) {
        if (pmat) matClose(pmat);
        if (pa) mxDestroyArray(pa);
        throw;
    }
}
