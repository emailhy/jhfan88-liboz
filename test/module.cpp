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
#include "module.h"
#include "paramui.h"


Module::Module() : QObject() {
    m_player = 0;
    m_dirty = true;
}


Module::~Module() {
}


void Module::saveSettings(QSettings& settings) {
    AbstractParam::saveSettings(settings, this);
}


void Module::restoreSettings(QSettings& settings) {
    AbstractParam::restoreSettings(settings, this);
}


QString Module::title() const {
    int index = metaObject()->indexOfClassInfo("title");
    if (index >= 0) return metaObject()->classInfo(index).value();
    return objectName();
}


QString Module::caption() const {
    int index = metaObject()->indexOfClassInfo("caption");
    if (index >= 0) return metaObject()->classInfo(index).value();
    return QString();
}


const QImage& Module::output() {
    check();
    return m_output;
}


void Module::setOutput(const QImage& image) {
    m_dirty = false;
    m_output = image;
    m_output.setText("Description", title() + "; " + AbstractParam::paramText(this));
    outputChanged(m_output);
}


void Module::process() {
    QImage err(256,256, QImage::Format_ARGB32);
    err.fill(Qt::red);
    setOutput(err);
}


void Module::check() {
    if (!isDirty()) return;
    qDebug() << "Processing" << objectName();
    try {
        process();
    }
    catch ( std::exception& e ) {
        qWarning() << e.what();
    }
}


QWidget* Module::createUI() {
    QWidget *w = new QWidget;
    QVBoxLayout *vbox = new QVBoxLayout(w);
    vbox->setContentsMargins(8,8,8,8);
    ParamUI *ui = new ParamUI(w, this);
    vbox->addWidget(ui);
    vbox->addStretch(0);
    return w;
}


void Module::setDirty() {
    if (!m_dirty) {
        m_dirty = true;
        dirty();
    }
}


ModulePlugin::ModulePlugin(const QMetaObject **pm) : m(pm) {
}


const QMetaObject** ModulePlugin::metaObjects() const {
    return m;
}

