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
#include "modulelist.h"
#include "moduledialog.h"
#include "paramui.h"


ModuleList::ModuleList(QWidget *parent) : QTabWidget(parent) {
    m_player = 0;

    setTabPosition(QTabWidget::West);
    setDocumentMode(false);

    QObjectList I = QPluginLoader::staticInstances();
    for (int i = 0; i < I.size(); ++i) {
        ModulePlugin* f = qobject_cast<ModulePlugin*>(I[i]);
        if (f) {
            const QMetaObject **p = f->metaObjects();
            while (*p) {
                m_metaObjects.insert((*p)->className(), *p);
                ++p;
            }
        }
    }

    connect(this, SIGNAL(currentChanged(int)), this, SIGNAL(listChanged()));
}


ModuleList::~ModuleList() {
}


void ModuleList::restoreSettings(QSettings& settings) {
    bool bs = blockSignals(true);

    int current = -1;
    QString currentModule = settings.value("currentModule").toString();
    QStringList M = settings.value("moduleOrder").toStringList();
    for (int i = 0; i < M.size(); ++i) {
        settings.beginGroup(M[i]);
        QString className = settings.value("className").toString();
        Module *m = create(className, M[i]);
        if (m) {
            m->restoreSettings(settings);

            settings.beginGroup("ui");
            bool enabled = settings.value("enabled", true).toBool();
            setTabEnabled(i, enabled);
            QScrollArea *sa = (QScrollArea*)m_widgets[i];
            ParamUI::restoreSettings(settings, sa->widget());
            settings.endGroup();

            settings.endGroup();

            if (currentModule == M[i]) current = i;
        }
    }
    if (current >= 0) setCurrentIndex(current);
    blockSignals(bs);

    if (m_modules.isEmpty()) {
        reset();
        return;
    }

    listChanged();
}


void ModuleList::saveSettings(QSettings& settings) {
    settings.remove("");

    if (current()) {
        settings.setValue("currentModule", current()->objectName());
    }
    QStringList order;
    for (int i = 0; i < m_modules.size(); ++i) {
        Module *m = m_modules[i];
        order.append(m->objectName());
        settings.beginGroup(m->objectName());
        settings.setValue("className", m->metaObject()->className());
        m->saveSettings(settings);

        QScrollArea *sa = m_widgets[i];
        settings.beginGroup("ui");
        settings.setValue("enabled", isTabEnabled(i));
        ParamUI::saveSettings(settings, sa->widget());
        settings.endGroup();

        settings.endGroup();
    }

    settings.setValue("moduleOrder", order);
}


Module* ModuleList::current() const {
    return m_modules.value(currentIndex());
}


QList<Module*> ModuleList::activeModules() const {
    QList<Module*> A;
    for (int i = 0; i < m_modules.size(); ++i) {
        if (isTabEnabled(i)) {
            A.append(m_modules[i]);
        }
    }
    return A;
}


void ModuleList::reset() {
    bool bs = blockSignals(true);

    qDeleteAll(m_modules);
    m_modules.clear();
    qDeleteAll(m_widgets);
    m_widgets.clear();

    QMap<QString,const QMetaObject*>::iterator i;
    for (i = m_metaObjects.begin(); i != m_metaObjects.end(); ++i) {
        const QMetaObject *mo = i.value();

        {
            bool startUp = false;
            int idx = mo->indexOfClassInfo("startUp");
            if (idx >= 0) {
                startUp = QVariant(mo->classInfo(idx).value()).toBool();
            }
            if (!startUp) continue;
        }

        QString name;
        {
            int idx = mo->indexOfClassInfo("name");
            if (idx >= 0)
                name = mo->classInfo(idx).value();
            else
                name = mo->className();
        }

        create(mo, name);
    }

    blockSignals(bs);
    listChanged();
}


void ModuleList::setCurrent(Module *m) {
    int index = m_modules.indexOf(m);
    if (currentIndex() != index) {
        setCurrentIndex(index);
    }
}


Module* ModuleList::create(const QMetaObject *mo, const QString& name) {
    Module *m = qobject_cast<Module*>(mo->newInstance());
    if (!m) {
        qCritical() << "Creating module" << mo->className() << "failed!";
        return 0;
    }
    m->setObjectName(name);
    m->setParent(this);
    m->setPlayer(m_player);
    connect(m, SIGNAL(dirty()), this, SLOT(handleDirty()));
    connect(m, SIGNAL(outputChanged(const QImage&)), this, SLOT(handleDirty()));
    m_modules.append(m);

    QScrollArea *sa = new QScrollArea();
    QWidget *w = m->createUI();
    sa->setWidget(w);
    sa->setFrameStyle(QFrame::NoFrame);
    sa->setFocusPolicy(Qt::NoFocus);
    sa->setWidgetResizable(true);
    addTab(sa, m->objectName());
    m_widgets.append(sa);

    return m;
}


Module* ModuleList::create(const QString& className, const QString& name) {
    const QMetaObject *mo = m_metaObjects.value(className);
    if (!mo) {
        qCritical() << "Creating module" << className << "failed!";
        return 0;
    }
    return create(mo, name);
}


void ModuleList::handleDirty() {
    Module *current = m_modules.value(currentIndex());
    if (current && (sender() == current)) {
        dirty();
    }
}


void ModuleList::edit(QWidget *parent, ModuleList *list) {
    ModuleDialog *d = new ModuleDialog(parent, list);
    d->exec();
    delete d;
}
