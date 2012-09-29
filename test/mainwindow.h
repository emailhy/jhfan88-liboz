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

#include "ui_mainwindow.h"
#include "videoplayer.h"
#include "modulelist.h"

class VideoPlayer;
class LogWindow;

class MainWindow : public QMainWindow, protected Ui_MainWindow {
    Q_OBJECT
public:
    MainWindow();
    ~MainWindow();

    void restoreSettings();
    void closeEvent(QCloseEvent *e);

    bool open(int index, const QString& filename);
    void open(int index);
    void close(int index);
    void sync();

    QString textForModule(const QString& id, Module *module);
    void setOutput(Module *module);

    Module* current() { return m_moduleList->current(); }

protected slots:
    void updateCurrentFrame();
    void updateModuleOutput();
    void setDirty();

    void on_menuFile_aboutToShow();
    void on_menuOpen_aboutToShow();
    void on_menuClose_aboutToShow();
    void on_actionOpen0_triggered();
    void on_actionOpen1_triggered();
    void on_actionOpen2_triggered();
    void on_actionOpen3_triggered();
    void on_actionClose0_triggered();
    void on_actionClose1_triggered();
    void on_actionClose2_triggered();
    void on_actionClose3_triggered();
    void on_actionSavePNG_triggered();
    void on_actionSavePDF_triggered();
    void on_actionSaveAll_triggered();
    void on_actionShowInfo_triggered();
    void on_actionBatch_triggered();
    void on_actionBatchEx_triggered();
    void on_actionBatch2_triggered();
    void on_actionRecord_triggered();
    void on_actionRecord2_triggered();
    void on_actionModules_triggered();
    void on_actionAbout_triggered();
    void on_actionSelectDevice_triggered();

protected:
    VideoPlayer *m_player[4];
    bool m_dirty;
    LogWindow *m_logWindow;

    QString m_filename[4];
    bool m_showFrame;
    QString m_frameHeader;
    QString m_frameFooter;
    QString m_globalHeader;
    QString m_globalFooter;
    QString m_batchFormat;
    bool m_recAutoSize;
    int m_recWidth;
    int m_recHeight;
    bool m_recAutoFps;
    double m_recFps;
    QString m_recOrient;
    int m_recBitRate;
    QImage m_titleImage;
    double m_titleDuration;
    QImage m_creditsImage;
    double m_creditsDuration;
};
