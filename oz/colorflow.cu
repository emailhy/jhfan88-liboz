//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// based on code from http://vision.middlebury.edu/flow
//
#include <oz/colorflow.h>
#include <oz/minmax.h>
#include <oz/color.h>
#include <oz/prctile.h>
#include <oz/transform.h>
#include <oz/colorflow_util.h>
#include <cfloat>


namespace {
    struct imp_magflow : public oz::unary_function<float2,float> {
        inline __device__ float operator()( float2 f ) const {
            return sqrtf(f.x * f.x + f.y * f.y);
        }
    };


    struct imp_colorflow : public oz::unary_function<float2,float3> {
        float maxrad_;

        imp_colorflow( float maxrad ) : maxrad_(maxrad) {}

        inline __device__ float3 operator()( float2 v ) const {
            float2 f = ::clamp(v / maxrad_, -1, 1);
            return oz::colorflow::value(f);
        }
    };
}


oz::gpu_image oz::magflow( const gpu_image& src ) {
    return transform(src, imp_magflow());
}


oz::gpu_image oz::colorflow( const gpu_image& src, float radius_max, bool robust ) {
    gpu_image tmp = src;
    if (radius_max == 0) {
        if (robust) {
            gpu_image mag = magflow(tmp);
            float mag95 = prctile(mag, 95);
            tmp = clamp(tmp, make_float2(mag95)*(-1), make_float2(mag95));
        }
        radius_max = max(magflow(tmp));
    }
    return transform(tmp, imp_colorflow(radius_max + FLT_EPSILON));
}


