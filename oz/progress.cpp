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
#include <oz/progress.h>
#include "qprogressdialog.h"
#include "qapplication.h"
#include <oz/math_util.h>
#include <vector>


static QProgressDialog* g_progress_dlg = 0;
static int g_progress_value = -1;
static std::vector<double4> g_progress_stack;


oz::progress_t::progress_t(double a, double b, double p, double q) {
    if (!g_progress_dlg) {
        g_progress_dlg = new QProgressDialog(qApp->applicationName(), "Cancel", 0, 100);
        g_progress_dlg->setMinimumDuration(1000);
        g_progress_value = 0;
    }
    g_progress_stack.push_back(make_double4(a, b, p, q));
}

oz::progress_t::~progress_t() {
    g_progress_stack.pop_back();
    if (g_progress_stack.empty() && g_progress_dlg) {
        delete g_progress_dlg;
        g_progress_dlg = 0;
    }
}


bool oz::progress_t::operator()(double t) {
    if (!g_progress_dlg) return false;

    double u = t;
    for (int i = (int)g_progress_stack.size() - 1; i >=0; --i) {
        double a =  g_progress_stack[i].x;
        double b =  g_progress_stack[i].y;
        double p =  g_progress_stack[i].z;
        double q =  g_progress_stack[i].w;
        u = (u - p) / (q - p);
        u = (1 - u) * a +  u * b;
    }
    int value = 100 * u;
    if (value != g_progress_value) {
        g_progress_value = value;
        g_progress_dlg->setValue(value);
        qApp->processEvents();
    }
    if (g_progress_dlg->wasCanceled()) {
        g_progress_dlg->deleteLater();
        g_progress_dlg = 0;
        return false;
    }
    return true;
}
