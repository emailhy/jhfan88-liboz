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
#include "moduledialog.h"


ModuleDialog::ModuleDialog(QWidget *parent, ModuleList *list)
    : QDialog(parent), m_list(list), m_lastIndex(-1)
{
    m_icons[0] = QIcon(":/images/qmark.png");
    m_icons[1] = QIcon(":/images/star1.png");
    m_icons[2] = QIcon(":/images/star2.png");
    m_icons[3] = QIcon(":/images/star3.png");
    m_icons[4] = QIcon(":/images/bug.png");
    m_icons[5] = QIcon(":/images/error.png");

    setupUi(this);
    new QShortcut(QKeySequence(Qt::Key_Insert), this, SLOT(addModule()));
    new QShortcut(QKeySequence(Qt::Key_Delete), this, SLOT(removeModule()));

    for (int i = 0; i < m_list->m_modules.size(); ++i) {
        Module *m = m_list->m_modules[i];
        const QMetaObject *mo = m->metaObject();

        QString description;
        int stars = 0;
        {
            int idx = mo->indexOfClassInfo("description");
            if (idx < 0) {
                idx = mo->indexOfClassInfo("title");
                if (idx < 0) {
                    idx = mo->indexOfClassInfo("caption");
                    if (idx < 0) {
                        idx = mo->indexOfClassInfo("name");
                    }
                }
            }
            if (idx >= 0)
                description = mo->classInfo(idx).value();
            else
                description = mo->className();
            description.replace('\n', "; ");
        }

        QTreeWidgetItem *li = new QTreeWidgetItem(m_treeWidget);
        li->setData(0, Qt::DisplayRole, m->objectName());
        li->setData(0, Qt::UserRole, i);
        li->setData(1, Qt::DisplayRole, description);

        li->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled |
                     Qt::ItemIsSelectable | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled);
        li->setCheckState(0, m_list->isTabEnabled(i)? Qt::Checked : Qt::Unchecked);
    }

    QTreeWidgetItem *root = m_treeWidget->invisibleRootItem();
    if (root->childCount() > 0) {
        m_treeWidget->setCurrentItem(root->child(0));
    }
    m_treeWidget->resizeColumnToContents(0);
    m_treeWidget->resizeColumnToContents(1);
}


void ModuleDialog::accept() {
    QTreeWidgetItem *root = m_treeWidget->invisibleRootItem();

    m_list->blockSignals(true);
    m_list->setUpdatesEnabled(false);

    QList<Module*> M = m_list->m_modules;
    QList<QScrollArea*> W = m_list->m_widgets;

    Module* current = m_list->current();
    m_list->clear();
    m_list->m_modules.clear();
    m_list->m_widgets.clear();

    for (int i = 0; i < root->childCount(); ++i) {
        QTreeWidgetItem *li = root->child(i);
        int x = li->data(0, Qt::UserRole).toInt();
        if (x >=0 ) {
            M[x]->setObjectName(li->data(0, Qt::DisplayRole).toString());
            m_list->m_modules.append(M[x]);
            m_list->m_widgets.append(W[x]);
            m_list->addTab(W[x], M[x]->objectName());
            M[x] = 0;
            W[x] = 0;
        } else {
            QString name = li->data(0, Qt::DisplayRole).toString();
            const QMetaObject *mo = (const QMetaObject*)(li->data(1, Qt::UserRole).value<void*>());
            m_list->create(mo, name);
        }
        m_list->setTabEnabled(i, li->checkState(0) == Qt::Checked);
    }

    for (int i = 0; i < M.size(); ++i) if (M[i]) delete(M[i]);
    for (int i = 0; i < W.size(); ++i) if (W[i]) delete(W[i]);

    {
        int index = m_list->m_modules.indexOf(current);
        if ((index >= 0) && m_list->isTabEnabled(index)) {
            m_list->setCurrentIndex(index);
        }
    }

    m_list->blockSignals(false);
    m_list->setUpdatesEnabled(true);

    m_list->listChanged();

    QDialog::accept();
}


void ModuleDialog::reset() {
    m_treeWidget->clear();

    QMap<QString,const QMetaObject*>::iterator i;
    for (i = m_list->m_metaObjects.begin(); i != m_list->m_metaObjects.end(); ++i) {
        const QMetaObject *mo = i.value();
        QString name;
        {
            int idx = mo->indexOfClassInfo("name");
            if (idx >= 0)
                name = mo->classInfo(idx).value();
            else
                name = mo->className();
        }

        bool active;
        {
            int idx = mo->indexOfClassInfo("startUp");
            active = (idx >= 0) && QVariant(mo->classInfo(idx).value()).toBool();
        }

        QTreeWidgetItem *ti = new QTreeWidgetItem(m_treeWidget);
        ti->setData(0, Qt::DisplayRole, makeUnique(name));
        ti->setData(0, Qt::UserRole, -1);
        ti->setData(1, Qt::DisplayRole, name);
        ti->setData(1, Qt::UserRole, QVariant::fromValue<void*>((void*)mo));
        ti->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        ti->setCheckState(0, active? Qt::Checked : Qt::Unchecked);
    }

    QTreeWidgetItem *top = m_treeWidget->topLevelItem(0);
    if (top) m_treeWidget->setCurrentItem(top);

    m_treeWidget->resizeColumnToContents(0);
    m_treeWidget->resizeColumnToContents(1);
}


void ModuleDialog::moveUp() {
    QList<QTreeWidgetItem*> S = m_treeWidget->selectedItems();
    if (!S.isEmpty()) {
        int index = m_treeWidget->indexOfTopLevelItem(S.first());
        if (index >= 1) {
            QTreeWidgetItem *currentItem = m_treeWidget->currentItem();
            for (int i = 0; i < S.size(); ++i) {
                m_treeWidget->takeTopLevelItem(index);
            }
            m_treeWidget->insertTopLevelItems(index - 1, S);
            m_treeWidget->setCurrentItem(currentItem, QItemSelectionModel::NoUpdate);
            for (int i = 0; i < S.size(); ++i) {
                S[i]->setSelected(true);
            }
        }
    }
}


void ModuleDialog::moveDown() {
    QList<QTreeWidgetItem*> S = m_treeWidget->selectedItems();
    if (!S.isEmpty()) {
        int index = m_treeWidget->indexOfTopLevelItem(S.first());
        if (index < m_treeWidget->topLevelItemCount() - S.size()) {
            QTreeWidgetItem *currentItem = m_treeWidget->currentItem();
            for (int i = 0; i < S.size(); ++i) {
                m_treeWidget->takeTopLevelItem(index);
            }
            m_treeWidget->insertTopLevelItems(index + 1, S);
            m_treeWidget->setCurrentItem(currentItem, QItemSelectionModel::NoUpdate);
            for (int i = 0; i < S.size(); ++i) {
                S[i]->setSelected(true);
            }
        }
    }
}


void ModuleDialog::addModule() {
    QDialog dlg(this);
    dlg.setWindowTitle("Add Module");
    dlg.setMinimumHeight(600);
    dlg.setMinimumWidth(800);
    QVBoxLayout *vbox = new QVBoxLayout(&dlg);
    QTreeWidget *tree = new QTreeWidget(&dlg);
    tree->setHeaderHidden(false);
    tree->setAllColumnsShowFocus(true);
    tree->setAlternatingRowColors(true);
    tree->setColumnCount(2);
    tree->setHeaderLabels(QStringList() << "Module" << "Description");
    tree->setIconSize(QSize(16,16));
    vbox->addWidget(tree);
    QDialogButtonBox *bbox = new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel,Qt::Horizontal, &dlg);
    vbox->addWidget(bbox);
    connect(bbox, SIGNAL(accepted()), &dlg, SLOT(accept()));
    connect(bbox, SIGNAL(rejected()), &dlg, SLOT(reject()));
    connect(tree, SIGNAL(itemDoubleClicked(QTreeWidgetItem*,int)), &dlg, SLOT(accept()));

    QHash<QString, QTreeWidgetItem*> H;
    QMap<QString,const QMetaObject*>::iterator i;
    for (i = m_list->m_metaObjects.begin(); i != m_list->m_metaObjects.end(); ++i) {
        const QMetaObject *mo = i.value();

        QStringList category;
        {
            QString c;
            int idx = mo->indexOfClassInfo("category");
            if (idx >= 0)
                c = mo->classInfo(idx).value();
            else
                c = "Unknown";
            category = c.split('/', QString::SkipEmptyParts);
        }

        QString c;
        QTreeWidgetItem *p = tree->invisibleRootItem();
        for (int j = 0; j < category.size(); ++j) {
            c += "/" + category[j];
            QTreeWidgetItem *q = H.value(c);
            if (!q) {
                q =  new QTreeWidgetItem(p, QStringList(category[j]));
                q->setExpanded(true);
                q->setFlags(q->flags() & ~Qt::ItemIsSelectable);
                H.insert(c, q);
            }
            p = q;
        }

        QString name;
        {
            int idx = mo->indexOfClassInfo("name");
            if (idx >= 0)
                name = mo->classInfo(idx).value();
            else
                name = mo->className();
        }

        QString description;
        {
            int idx = mo->indexOfClassInfo("description");
            if (idx >= 0) description = mo->classInfo(idx).value();
            else {
                idx = mo->indexOfClassInfo("title");
                if (idx >= 0) description = mo->classInfo(idx).value();
                else {
                    idx = mo->indexOfClassInfo("caption");
                    if (idx >= 0) description = mo->classInfo(idx).value();
                }
            }
            description.replace('\n', "; ");
        }

        QTreeWidgetItem *item = new QTreeWidgetItem(p, QStringList() << name << description);
        item->setData(0, Qt::UserRole, QVariant::fromValue<void*>((void*)mo));

        int idx = mo->indexOfClassInfo("rating");
        if (idx >= 0) {
            QString s = mo->classInfo(idx).value();
            if (s == "*") item->setIcon(0, m_icons[1]);
            else if (s == "**") item->setIcon(0, m_icons[2]);
            else if (s == "***") item->setIcon(0, m_icons[3]);
            else if (s == "debug") item->setIcon(0, m_icons[4]);
            else if (s == "!") item->setIcon(0, m_icons[5]);
            else item->setIcon(0, m_icons[0]);
        } else {
            item->setIcon(0, m_icons[0]);
        }
    }
    tree->sortItems(0, Qt::DescendingOrder);
    tree->resizeColumnToContents(0);
    tree->resizeColumnToContents(1);
    tree->resizeColumnToContents(2);
    if (dlg.exec() == QDialog::Accepted) {
        QList<QTreeWidgetItem*> s = tree->selectedItems();
        if (!s.isEmpty()) {
            QTreeWidgetItem *a =s.first();
            const QMetaObject *mo = (const QMetaObject*)(a->data(0, Qt::UserRole).value<void*>());
            if (mo) {
                QString name = a->data(0, Qt::DisplayRole).toString();

                QTreeWidgetItem *c = m_treeWidget->currentItem();
                int index = m_treeWidget->indexOfTopLevelItem(c) + 1;

                QTreeWidgetItem *ti = new QTreeWidgetItem;
                ti->setData(0, Qt::DisplayRole, makeUnique(name));
                ti->setData(0, Qt::UserRole, -1);
                ti->setData(1, Qt::DisplayRole, name);
                ti->setData(1, Qt::UserRole, QVariant::fromValue<void*>((void*)mo));
                ti->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
                ti->setCheckState(0, Qt::Checked);

                m_treeWidget->insertTopLevelItem(index, ti);
                m_treeWidget->setCurrentItem(ti);
            }
        }
    }
}


void ModuleDialog::removeModule() {
    QList<QTreeWidgetItem*> S = m_treeWidget->selectedItems();
    if (!S.isEmpty()) {
        for (int i = 0; i < S.size(); ++i) {
            delete S[i];
        }
        QTreeWidgetItem *currentItem = m_treeWidget->currentItem();
        if (currentItem) {
            currentItem->setSelected(true);
        }
    }
}


void ModuleDialog::selectionChanged() {
    QList<QTreeWidgetItem*> S = m_treeWidget->selectedItems();
    m_deleteButton->setEnabled(!S.isEmpty());
    if (!S.isEmpty()) {
        m_upButton->setEnabled(m_treeWidget->indexOfTopLevelItem(S.first()) > 0);
        m_downButton->setEnabled(m_treeWidget->indexOfTopLevelItem(S.last()) < m_treeWidget->topLevelItemCount() - 1);
    }
}


void ModuleDialog::editItem(QTreeWidgetItem *item, int column) {
    QString name = item->data(0, Qt::DisplayRole).toString();
    bool ok;
    QString newName = QInputDialog::getText(this, "Edit", "Name:", QLineEdit::Normal, name, &ok);
    if (ok && (newName != name)) {
        name = makeUnique(newName);
        item->setData(0, Qt::DisplayRole, newName);
    }
}


QString ModuleDialog::makeUnique(const QString& name) {
    QSet<QString> H;
    for (int i = 0; i < m_treeWidget->topLevelItemCount(); ++i) {
        QTreeWidgetItem *item = m_treeWidget->topLevelItem(i);
        H.insert(item->data(0, Qt::DisplayRole).toString());
    }
    if (!H.contains(name)) return name;
    QString nn;
    int N = 2;
    do {
        nn = QString("%1-%2").arg(name).arg(N++);
    } while (H.contains(nn));
    return nn;
}
