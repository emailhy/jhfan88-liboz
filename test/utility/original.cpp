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
#include "original.h"
#include <oz/hist.h>


Original::Original() {
    new ParamInt(this, "input_index", 0, 0, 3, 1, &m_inputIndex);

    ParamGroup *g;
    g = new ParamGroup(this, "auto_levels", false, &auto_levels);
    new ParamDouble(g, "threshold", 0.1, 0, 100, 0.05, &auto_levels_threshold);
}


void Original::process() {
    QImage I;
    VideoPlayer *p = player(m_inputIndex);
    if (p) I = p->image();

    if (auto_levels && !I.isNull()) {
        oz::gpu_image src = oz::from_qimage(I).convert(oz::FMT_FLOAT3);
        src = oz::hist_auto_levels(src, auto_levels_threshold);
        setOutput(oz::to_qimage(src));
    } else {
        setOutput(I);
    }
}


void Original::draw(ImageView *view, QPainter &p, int pass) {
    Module::draw(view, p, pass);
}


OriginalEx::OriginalEx() {
    new ParamInt(this, "number", 1, 0, 100, 1, &m_number);
}


void OriginalEx::process() {
    QRegExp rx("-0\\.");
    QString fn = m_player->filename();
    fn.replace(rx, QString("-%1.").arg(m_number));

    QFileInfo fi(fn);
    QFileInfo ff(fi.dir(), "info.ini");
    qDebug() << ff.absoluteFilePath();

    m_title = "N/A";
    if (ff.exists()) {
        QSettings settings(ff.absoluteFilePath(), QSettings::IniFormat);
        m_title = settings.value("title/" + fi.fileName()).toString();
        if (m_title.isEmpty()) {
            m_title = settings.value("title", "N/A").toString();
        }
    }

    QImage image(fn);
    setOutput(image);
}


QString OriginalEx::title() const {
    return m_title;
}
