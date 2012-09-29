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
#include "propertyui.h"
#include <cfloat>


PropertyUI::PropertyUI(QWidget *parent, QObject *object) : RolloutBox(parent), m_object(object) {
    connect(object, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));

    const QMetaObject *meta = object->metaObject();
    QList<QByteArray> pnames;
    
    for (int i = 0/*meta->propertyOffset()*/; i < meta->propertyCount(); ++i) {
        QMetaProperty mp = meta->property(i);

        if (!mp.hasNotifySignal()) 
            continue;
        QMetaMethod mm = mp.notifySignal();
        QList<QByteArray> pn = mm.parameterTypes();
        if ((pn.size() != 1) || (pn[0] != mp.typeName()))
            continue;
        QByteArray notifySignature(1, '0' + QSIGNAL_CODE);
        notifySignature.append(mm.signature());

        switch (mp.type()) {
            case QVariant::Bool: {
                QCheckBox *w = new QCheckBox(this);
                w->setObjectName(mp.name());
                w->setFixedHeight(fontMetrics().height()+6);
                w->setChecked(object->property(mp.name()).toBool());
                QLabel *l = new QLabel(mp.name(), this);
                l->setFixedHeight(fontMetrics().height()+6);
                l->setBuddy(w);
                addWidget(l,w);
                connect(w, SIGNAL(toggled(bool)), this, SLOT(setProperty(bool)));
                connect(object, notifySignature.data(), w, SLOT(setChecked(bool)));
                break;
            }

            case QVariant::Int: {
                if (!mp.isEnumType()) {
                    QSpinBox *w = new QSpinBox(this);
                    w->setObjectName(mp.name());
                    w->setFixedHeight(fontMetrics().height()+6);
                    w->setRange(-INT_MAX, INT_MAX);
                    w->setValue(object->property(mp.name()).toInt());
                    w->setKeyboardTracking(false);
                    QLabel *l = new QLabel(mp.name(), this);
                    l->setBuddy(w);
                    addWidget(l,w);
                    connect(w, SIGNAL(valueChanged(int)), this, SLOT(setProperty(int)));
                    connect(object, notifySignature.data(), w, SLOT(setValue(int)));
                } else {
                    QComboBox *w = new PropertyEnumBox(this, object, mp.name());
                    w->setFixedHeight(fontMetrics().height()+6);
                    const QMetaEnum me = mp.enumerator();
                    for (int i = 0; i < me.keyCount(); ++i) {
                        w->addItem(me.key(i), me.value(i));
                    }
                    QLabel *l = new QLabel(mp.name(), this);
                    l->setBuddy(w);
                    addWidget(l,w);
                    connect(w, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(setProperty(const QString&)));
                    connect(object, notifySignature.data(), w, SLOT(updateValue()));
                }
                break;
            }

            case QVariant::Double: {
                QDoubleSpinBox *w = new QDoubleSpinBox(this);
                w->setObjectName(mp.name());
                w->setFixedHeight(fontMetrics().height()+6);
                w->setDecimals(5);
                w->setRange(-FLT_MAX, FLT_MAX);
                w->setValue(object->property(mp.name()).toDouble());
                w->setKeyboardTracking(false);
                QLabel *l = new QLabel(mp.name(), this);
                l->setBuddy(w);
                addWidget(l,w);
                connect(w, SIGNAL(valueChanged(double)), this, SLOT(setProperty(double)));
                connect(object, notifySignature.data(), w, SLOT(setValue(double)));
                break;
            }

            default:
                break;
        }
    }
}


void PropertyUI::setProperty(bool value) {
    QString name = sender()->objectName();
    if (!name.isEmpty()) {
        m_object->setProperty(name.toLocal8Bit().data(), QVariant(value));
    }
}


void PropertyUI::setProperty(int value) {
    QString name = sender()->objectName();
    if (!name.isEmpty()) {
        m_object->setProperty(name.toLocal8Bit().data(), QVariant(value));
    }
}


void PropertyUI::setProperty(double value) {
    QString name = sender()->objectName();
    if (!name.isEmpty()) {
        m_object->setProperty(name.toLocal8Bit().data(), QVariant(value));
    }
}


void PropertyUI::setProperty(const QString& value) {
    QString name = sender()->objectName();
    if (!name.isEmpty()) {
        m_object->setProperty(name.toLocal8Bit().data(), QVariant(value));
    }
}


PropertyEnumBox::PropertyEnumBox(QWidget *parent, QObject *object, const QString& name) 
    : QComboBox(parent), m_object(object)
{
    setObjectName(name);
    updateValue();
}



void PropertyEnumBox::updateValue() {
    QVariant value = m_object->property(objectName().toLatin1().data());
    int index = findData(value);
    setCurrentIndex(index);
}
