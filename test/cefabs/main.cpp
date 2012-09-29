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
#include "flowshock.h"
#include "ivacef.h"
#include "ssia.h"


static const QMetaObject* module_cefabs[] = {
    &SSIA::staticMetaObject,
    &SSIA2::staticMetaObject,
    &FlowShock::staticMetaObject,
    &IVACEF::staticMetaObject,
    NULL
};


Q_EXPORT_PLUGIN2(module_cefabs, ModulePlugin(module_cefabs));
Q_IMPORT_PLUGIN(module_cefabs)
