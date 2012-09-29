// by Jan Eric Kyprianidis <www.kyprianidis.com>
//#include "oilpaintcs5.h"
//#include "oilpaintcs6.h"
#include "oilpaint.h"
#include "watercolor.h"


static const QMetaObject* module_artistic[] = {
    //&OilPaintCS6::staticMetaObject,
    //&OilPaintCS5::staticMetaObject,
    &OilPaint::staticMetaObject,
    &Watercolor::staticMetaObject,
    NULL
};


Q_EXPORT_PLUGIN2(module_artistic, ModulePlugin(module_artistic));
Q_IMPORT_PLUGIN(module_artistic)
