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

#include "module.h"

class ModuleList : public QTabWidget {
    Q_OBJECT
public:
    ModuleList(QWidget *parent);
    ~ModuleList();

    void setPlayer(VideoPlayer *player) { m_player = player; }
    VideoPlayer* player() const { return m_player; }

    void restoreSettings(QSettings& settings);
    void saveSettings(QSettings& settings);

    Module* current() const;
    QList<Module*> modules() const { return m_modules; }
    QList<Module*> activeModules() const;

public slots:
    void setCurrent(Module *m);
    void reset();

signals:
    void listChanged();
    void dirty();

public:
    static void edit(QWidget *parent, ModuleList *list);

protected slots:
    void handleDirty();

protected:
    Module* create(const QMetaObject *mo, const QString& name);
    Module* create(const QString& className, const QString& name);

    QMap<QString,const QMetaObject*> m_metaObjects;
    QList<Module*> m_modules;
    QList<QScrollArea*> m_widgets;
    VideoPlayer *m_player;

    friend class ModuleDialog;
};
