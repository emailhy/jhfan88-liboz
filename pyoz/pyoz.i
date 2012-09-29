%module pyoz

%{
#define SWIG_FILE_WITH_INIT
#include <oz/gpu_image.h>
#include <oz/gauss.h>
#include <oz/color.h>
#include <oz/noise.h>
#include <oz/io.h>
#include <oz/hist.h>
#include <oz/st.h>
#include "oz/shuffle.h"
#include "oz/make.h"
#include <oz/cairo_support.h>
#include "numpy.h"
#define PY_ARRAY_UNIQUE_SYMBOL oz_ARRAY_API
#include <numpy/arrayobject.h>
#include <pycairo.h>
static Pycairo_CAPI_t *Pycairo_CAPI;
%}

%feature("compactdefaultargs");

%typemap(typecheck,precedence=10000) cairo_t* {
	$1 = PyObject_TypeCheck($input, &PycairoContext_Type);
}

%typemap(in) cairo_t* {
    $1 = 0;
    if (PyObject_TypeCheck($input, &PycairoContext_Type)) {
        $1 = ((PycairoContext*)$input)->ctx;
    } else {
        SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument " "$argnum"" of type '" "$type""'");
    }
}

%typemap(out) cairo_t* {
    $result = PycairoContext_FromContext($1, &PycairoContext_Type, NULL);
}

%typemap(typecheck,precedence=10000) cairo_surface_t* {
	$1 = PyObject_TypeCheck($input, &PycairoSurface_Type);
}

%typemap(in) cairo_surface_t* {
    $1 = 0;
    if (PyObject_TypeCheck($input, &PycairoSurface_Type)) {
        $1 = ((PycairoSurface*)$input)->surface;
    } else {
        SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument " "$argnum"" of type '" "$type""'");
    }
}

%typemap(out) cairo_surface_t* {
    $result = PycairoSurface_FromSurface($1, NULL);
}

%include "std_vector.i"
namespace std {
   %template(vectori) vector<int>;
   %template(vectorf) vector<float>;
};

%include exception.i
%exception {
    try {
        $action
    }
    catch (std::exception &e) {
        SWIG_exception(SWIG_RuntimeError, const_cast<char *>(e.what()));
    }
}

%init %{
    import_array();
    Pycairo_IMPORT;
%}


%ignore oz::image_format_type_size;
%ignore oz::image_format_str;
%ignore oz::image_format_invalid_msg;
%include "oz/image_format.h"

%rename(get_w) oz::gpu_image::w() const;
%rename(get_h) oz::gpu_image::h() const;
//%feature("ref") oz::gpu_image "$this->add_ref();"
//%feature("unref") oz::gpu_image "$this->unref();"

%rename(get_w) oz::cpu_image::w() const;
%rename(get_h) oz::cpu_image::h() const;

namespace oz {
    class gpu_image {
    public:
        gpu_image();
        gpu_image( unsigned w, unsigned h, image_format_t format );
        unsigned w() const;
        unsigned h() const;

        %pythoncode %{
            __swig_getmethods__["w"] = get_w
            if _newclass: w = property(get_w)
            __swig_getmethods__["h"] = get_h
            if _newclass: h = property(get_h)
        %}

        gpu_image clone() const;
        oz::cpu_image cpu() const;
        gpu_image convert( image_format_t format, bool clone=false ) const;
    };

    gpu_image adjust( const gpu_image& src, float a, float b );
    gpu_image invert( const gpu_image& src );
    gpu_image saturate( const gpu_image& src );
    gpu_image clamp( const gpu_image& src, float a, float b );
    gpu_image clamp( const gpu_image& src, float2 a, float2 b );
    gpu_image clamp( const gpu_image& src, float3 a, float3 b );
    gpu_image clamp( const gpu_image& src, float4 a, float4 b );
    gpu_image lerp( const gpu_image& src0,const gpu_image& src1, float t );
    gpu_image abs( const gpu_image& src );
    gpu_image sqrt( const gpu_image& src );
	gpu_image pow( const gpu_image& src, float y );
    gpu_image abs_diff( const gpu_image& src0, const gpu_image& src1 );
    gpu_image log_abs( const gpu_image& src );

    class cpu_image {
    public:
        cpu_image();
        cpu_image( unsigned w, unsigned h, image_format_t format );
        unsigned w() const;
        unsigned h() const;

        %pythoncode %{
            __swig_getmethods__["w"] = get_w
            if _newclass: w = property(get_w)
            __swig_getmethods__["h"] = get_h
            if _newclass: h = property(get_h)
        %}

        cpu_image clone() const;
        oz::gpu_image gpu() const;
        cpu_image convert( image_format_t format, bool clone=false ) const;
    };
}

%include "numpy.h"
%include "oz/gauss.h"
%include "oz/io.h"
%include "oz/color.h"
%include "oz/noise.h"
%include "oz/hist.h"
%include "oz/st.h"
%include "oz/shuffle.h"
%include "oz/make.h"
%include "oz/cairo_support.h"
