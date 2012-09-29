//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2011-2012 Computer Graphics Systems Group at the
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
#include "numpy.h"
#define PY_ARRAY_UNIQUE_SYMBOL oz_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>
#include <oz/type_traits.h>


template<typename T>
static PyObject* to_pyarray_T( const oz::cpu_image& src, int typenum ) {
    npy_intp dims[3];
    dims[0] = src.h();
    dims[1] = src.w();
    dims[2] = oz::type_traits<T>::N;

    PyObject* o = PyArray_SimpleNew((dims[2] == 1)? 2 : 3, dims, typenum);
    npy_intp* strides = PyArray_STRIDES(o);
    T* data = (T*)PyArray_DATA(o);

    src.get(data, (unsigned)strides[0]);
    return o;
}


PyObject* oz::to_pyarray( const cpu_image& src ) {
    switch (src.format()) {
        case FMT_UCHAR:  return to_pyarray_T<uchar >(src, NPY_UBYTE);
        case FMT_UCHAR2: return to_pyarray_T<uchar2>(src, NPY_UBYTE);
        case FMT_UCHAR3: return to_pyarray_T<uchar3>(src, NPY_UBYTE);
        case FMT_UCHAR4: return to_pyarray_T<uchar4>(src, NPY_UBYTE);
        case FMT_FLOAT:  return to_pyarray_T<float >(src, NPY_FLOAT);
        case FMT_FLOAT2: return to_pyarray_T<float2>(src, NPY_FLOAT);
        case FMT_FLOAT3: return to_pyarray_T<float3>(src, NPY_FLOAT);
        case FMT_FLOAT4: return to_pyarray_T<float4>(src, NPY_FLOAT);
        default:
            OZ_INVALID_FORMAT();
    }
    return NULL;
}


PyObject* oz::to_pyarray( const gpu_image& src ) {
    return to_pyarray(src.cpu());
}


oz::cpu_image oz::from_pyarray( const PyObject *o ) {
    if (!PyArray_Check(o))
        OZ_X() << "Error: Object is not a numpy array!";

    int array_ndim = PyArray_NDIM(o);
    if ((array_ndim < 2) || (array_ndim > 3)) {
        OZ_X() << "Error: Unsupported array dimension: " << array_ndim;
    }

    const npy_intp* array_dims = PyArray_DIMS(o);
    const npy_intp* array_strides = PyArray_STRIDES(o);
    int array_type = PyArray_TYPE(o);
    void* array_data = PyArray_DATA(o);

    if (array_strides[0] < array_strides[1]) {
        OZ_X() << "Unsupported stride format. Only CORDER format is supported!";
    }

    if (array_ndim == 3) {
        if ((array_dims[2] < 2) || (array_dims[2] > 4)) {
            OZ_X() << "Error: Unsupported array dimension: "
                   << (unsigned)array_dims[0] << "x" << (unsigned)array_dims[1] << "x" << (unsigned)array_dims[2];
        }
    }

    int ndim = (array_ndim == 2)? 1 : (int)array_dims[2];
    unsigned w = (unsigned)array_dims[1];
    unsigned h = (unsigned)array_dims[0];
    unsigned pitch = (unsigned)array_strides[0];

    switch (array_type) {
        case NPY_UBYTE: {
            switch (ndim) {
                case 1: return cpu_image((uchar*)array_data, pitch, w, h);
                case 2: return cpu_image((uchar2*)array_data, pitch, w, h);
                case 3: return cpu_image((uchar3*)array_data, pitch, w, h);
                case 4: return cpu_image((uchar4*)array_data, pitch, w, h, false);
            }
            break;
        }

        case NPY_FLOAT: {
            switch (ndim) {
                case 1: return cpu_image((float*)array_data, pitch, w, h);
                case 2: return cpu_image((float2*)array_data, pitch, w, h);
                case 3: return cpu_image((float3*)array_data, pitch, w, h);
                case 4: return cpu_image((float4*)array_data, pitch, w, h, false);
            }
            break;
        }
    }
    OZ_X() << "Error: Unsupported array type!";
}
