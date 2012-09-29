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
#pragma once

#include <oz/gpu_image.h>
#include "imageview2.h"
#include "param.h"
#include "videoplayer.h"

class Module : public QObject, public ImageView::Handler {
    Q_OBJECT
public:
    Module();
    ~Module();

    void setPlayer(VideoPlayer *player) { m_player = player; }
    VideoPlayer* player(int num=0) const { return (num == 0)? m_player : m_player->slave(num - 1); }

    bool isDirty() const { return m_dirty; }
    const QImage& output();

    virtual void saveSettings(QSettings& settings);
    virtual void restoreSettings(QSettings& settings);
    virtual QString title() const;
    virtual QString caption() const;
    virtual QWidget* createUI();

public slots:
    void setDirty();

signals:
    void dirty();
    void outputChanged(const QImage& image);

protected:
    void setOutput(const QImage& image);
    virtual void process();
    virtual void check();

protected:
    VideoPlayer *m_player;
    QImage m_output;
    bool m_dirty;
};


class ModulePlugin : public QObject {
    Q_OBJECT
public:
    ModulePlugin(const QMetaObject **pm);
    const QMetaObject** metaObjects() const;
private:
    const QMetaObject **m;
};
