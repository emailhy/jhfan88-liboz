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
#include "testring.h"
#include "testpattern.h"
#include "ststat.h"
#include "gradtest.h"
#include "stinterp.h"
#include "stinterp2.h"


static const QMetaObject* module_st[] = {
    &TestRing::staticMetaObject,
    &TestPattern::staticMetaObject,
    &StStat::staticMetaObject,
    &GradTest::staticMetaObject,
    &StInterp::staticMetaObject,
    &StInterp2::staticMetaObject,
    NULL
};


Q_EXPORT_PLUGIN2(module_st, ModulePlugin(module_st));
Q_IMPORT_PLUGIN(module_st)
