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

#include "modulelist.h"
#include "ui_moduledialog.h"

class ModuleDialog : public QDialog, private Ui_ModuleDialog {
    Q_OBJECT
public:
    ModuleDialog(QWidget *parent, ModuleList *list);

public slots:
    virtual void accept();

protected slots:
    void reset();
    void moveUp();
    void moveDown();
    void addModule();
    void removeModule();
    void selectionChanged();
    void editItem(QTreeWidgetItem *item, int column);

protected:
    QString makeUnique(const QString& name);
    ModuleList *m_list;
    int m_lastIndex;
    QIcon m_icons[6];
};
