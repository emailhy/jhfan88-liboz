//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2011 Computer Graphics Systems Group at the
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

#include "rolloutbox.h"

class PropertyUI : public RolloutBox {
    Q_OBJECT
public:
    PropertyUI(QWidget *parent, QObject *object);

private slots:
    void setProperty(bool value);
    void setProperty(int value);
    void setProperty(double value);
    void setProperty(const QString& value);

private:
    QPointer<QObject> m_object;
};

class PropertyEnumBox : public QComboBox {
    Q_OBJECT
public:            
    PropertyEnumBox(QWidget *parent, QObject *object, const QString& name);

protected slots:
    void updateValue();

protected:
    QPointer<QObject> m_object;
};

