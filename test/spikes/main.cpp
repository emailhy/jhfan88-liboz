// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "reductiontest.h"
#include "noise.h"
#include "filtertest.h"
#include "resize.h"
#include "resample.h"
#include "variance.h"
#include "tvl1flow.h"
#include "otsu.h"
#include "cairotest.h"
#include "laplace_eq_test.h"
#include "convpyr_test.h"
#include "ffttest.h"
#include "fftconv.h"


static const QMetaObject* module_spikes[] = {
    &ReductionTest::staticMetaObject,
    &FilterTest::staticMetaObject,
    &NoiseTest::staticMetaObject,
    &ResizeTest::staticMetaObject,
    &ResampleTest::staticMetaObject,
    &VarianceTest::staticMetaObject,
    &TVL1Flow::staticMetaObject,
    &OtsuTest::staticMetaObject,
    &CairoTest::staticMetaObject,
    &LaplaceEqTest::staticMetaObject,
    &ConvPyrTest::staticMetaObject,
    &FFTTest::staticMetaObject,
    &FFTConvTest::staticMetaObject,
    NULL
};


Q_EXPORT_PLUGIN2(module_spikes, ModulePlugin(module_spikes));
Q_IMPORT_PLUGIN(module_spikes)
