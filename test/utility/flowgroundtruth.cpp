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
#include "flowgroundtruth.h"
#include <oz/flowio.h>
#include <oz/colorflow.h>
using namespace oz;


FlowGroundTruth::FlowGroundTruth() {
}


void FlowGroundTruth::process() {
    QString fn = m_player->filename();
    QFileInfo fi(fn);
    QFileInfo ff(fi.dir(), "flow10.flo");

    if (ff.exists()) {
        cpu_image cf = read_flow(ff.canonicalFilePath().toLocal8Bit().data());
        if (cf.is_valid()) {
            publish("groundtruth", colorflow(cf.gpu()));
        }
    } else {
        QImage src = player()->image();
        QImage r(src.width(), src.height(), QImage::Format_ARGB32);
        r.fill(0);
        publish("groundtruth", r);
    }
}
