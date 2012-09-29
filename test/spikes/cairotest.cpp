// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "cairotest.h"
#include <oz/color.h>
#include <oz/threshold.h>
#include <oz/cairo_support.h>
using namespace oz;


CairoTest::CairoTest() {
}


void CairoTest::process() {
    cpu_image src = cpuInput0();
    publish("src0", src);

    cairo_surface_t *surface = create_surface(src);
    cairo_t *cr = cairo_create(surface);
    cairo_surface_destroy(surface);

    cairo_set_line_width (cr, 3);
    cairo_set_source_rgb (cr, 0, 0, 0);
    cairo_rectangle(cr, 10, 10, 100, 100);
    cairo_stroke (cr);
    
    cairo_destroy(cr);

    publish("src1", src);
}
