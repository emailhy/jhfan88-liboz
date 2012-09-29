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
#include "mainwindow.h"
#include "logwindow.h"
#include <oz/ffmpeg_encoder.h>
#include "cudadevicedialog.h"
#include "propertyui.h"
#include "paramui.h"
#include <oz/qimage.h>
#include <oz/gpu_cache.h>


MainWindow::MainWindow() {
    m_player[0] = new VideoPlayer(this, ":/test.png", 16);
    for (int i = 1; i < 4; ++i) m_player[i] = new VideoPlayer(m_player[0], QString::null, 16);
    m_showFrame = false;
    m_dirty = true;
    setupUi(this);
    m_moduleList->setPlayer(m_player[0]);

    actionRecord->setEnabled(false);
    actionRecord2->setEnabled(false);

    m_splitter->setStretchFactor(0,0);
    m_splitter->setStretchFactor(1,100);
    m_splitter->setStretchFactor(2,0);
    m_logWindow = new LogWindow(this);
    m_logWindow->setVisible(false);
    //m_logWindow->setFixedHeight(550);
    verticalLayout->addWidget(m_logWindow);
    m_imageView->setFocus();

    m_videoControls->setAutoHide(true);
    connect(m_videoControls, SIGNAL(stepForward()), m_player[0], SLOT(stepForward()));
    connect(m_videoControls, SIGNAL(stepBack()), m_player[0], SLOT(stepBack()));
    connect(m_videoControls, SIGNAL(currentFrameTracked(int)), m_player[0], SLOT(setCurrentFrame(int)));
    connect(m_videoControls, SIGNAL(playbackChanged(bool)), m_player[0], SLOT(setPlayback(bool)));

    connect(m_player[0], SIGNAL(videoChanged(int)), m_videoControls, SLOT(setFrameCount(int)));
    connect(m_player[0], SIGNAL(playbackChanged(bool)), m_videoControls, SLOT(setPlayback(bool)));
    connect(m_player[0], SIGNAL(currentFrameChanged(int)), m_videoControls, SLOT(setCurrentFrame(int)));
    connect(m_player[0], SIGNAL(currentFrameChanged(int)), this, SLOT(updateCurrentFrame()));

    connect(m_moduleList, SIGNAL(listChanged()), this, SLOT(setDirty()));
    connect(m_moduleList, SIGNAL(dirty()), this, SLOT(setDirty()));

    ParamGroup *p;
    p = new ParamGroup(this, "show frame", false, &m_showFrame);
    new ParamChoice(p, "header", "off", "off|title|caption|title+caption|header|footer", &m_frameHeader);
    new ParamChoice(p, "footer", "off", "off|title|caption|title+caption|header|footer", &m_frameFooter);
    new ParamString(p, "header_text", "", true, &m_globalHeader);
    new ParamString(p, "footer_text", "", true, &m_globalFooter);

    p = new ParamGroup(this, "batch processing");
    new ParamChoice(p, "format", "png", "png|jpg", &m_batchFormat);

    p = new ParamGroup(this, "recording");
    new ParamInt(p, "bitrate", 8000000, 1, 10000000, 100000, &m_recBitRate);
    new ParamBool(p, "automatic size", true, &m_recAutoSize);
    new ParamInt(p, "width", 1280, 1, 4096, 1, &m_recWidth);
    new ParamInt(p, "height", 720, 1, 4096, 1, &m_recHeight);
    new ParamBool(p, "automatic fps", true, &m_recAutoFps);
    new ParamDouble(p, "fps", 25, 1, 100, 1, &m_recFps);
    new ParamChoice(p, "oritentation", "horizontal", "horizontal|vertical", &m_recOrient);
    new ParamImage(p, "title", QImage(), &m_titleImage);
    new ParamDouble(p, "title_duration", 2, 1, 10, 1, &m_titleDuration);
    new ParamImage(p, "credits", QImage(), &m_creditsImage);
    new ParamDouble(p, "credits_duration", 2, 1, 10, 1, &m_creditsDuration);

    {
        QWidget *w = new QWidget();
        QVBoxLayout *vbox = new QVBoxLayout(w);
        vbox->setContentsMargins(0,0,0,0);
        vbox->setSpacing(4);
        PropertyUI *pp = new PropertyUI(w, m_imageView);
        pp->setTitle("viewport");
        pp->setObjectName("viewport");
        vbox->addWidget(pp);
        ParamUI *pm = new ParamUI(w, this);
        vbox->addWidget(pm);
        vbox->addStretch();
        m_options->setWidget(w);
        m_options->setVisible(false);
    }

    connect(actionLog, SIGNAL(toggled(bool)), m_logWindow, SLOT(setVisible(bool)));
    connect(actionOptions, SIGNAL(toggled(bool)), m_options, SLOT(setVisible(bool)));
}


MainWindow::~MainWindow() {
}


void MainWindow::restoreSettings() {
    QSettings settings;
    restoreGeometry(settings.value("mainWindow/geometry").toByteArray());
    restoreState(settings.value("mainWindow/windowState").toByteArray());

    settings.beginGroup("imageView");
    m_imageView->restoreSettings(settings);
    settings.endGroup();

    settings.beginGroup("options");
    AbstractParam::restoreSettings(settings, this);
    settings.endGroup();
    for (int i = 0; i < 4; ++i) {
        m_filename[i] = settings.value(QString("filename%1").arg(i), "").toString();
    }

    settings.beginGroup("modules");
    m_moduleList->restoreSettings(settings);
    settings.endGroup();

    settings.beginGroup("options_ui");
    ParamUI::restoreSettings(settings, m_options);
    settings.endGroup();

    QApplication::processEvents();
    for (int i = 0; i < 4; ++i) {
        if (QFile::exists(m_filename[i])) {
            open(i, m_filename[i]);
        }
    }

    actionLog->setChecked(settings.value("showLog", false).toBool());
    actionOptions->setChecked(settings.value("showOptions", false).toBool());

    updateModuleOutput();
}


void MainWindow::closeEvent(QCloseEvent *e) {
    QSettings settings;
    settings.setValue("mainWindow/geometry", saveGeometry());
    settings.setValue("mainWindow/windowState", saveState());
    settings.setValue("showLog", actionLog->isChecked());
    settings.setValue("showOptions", actionOptions->isChecked());

    settings.beginGroup("options");
    AbstractParam::saveSettings(settings, this);
    settings.endGroup();
    for (int i = 0; i < 4; ++i) {
        settings.setValue(QString("filename%1").arg(i),
        m_player[i]->isValid()? m_filename[i] : "");
    }

    settings.beginGroup("imageView");
    m_imageView->saveSettings(settings);
    settings.endGroup();

    settings.beginGroup("modules");
    m_moduleList->saveSettings(settings);
    settings.endGroup();

    settings.beginGroup("options_ui");
    ParamUI::saveSettings(settings, m_options);
    settings.endGroup();

    QMainWindow::closeEvent(e);
}


bool MainWindow::open(int index, const QString& filename) {
    oz::gpu_cache_clear();
    if (!m_player[index]->open(filename)) {
        QMessageBox::critical(this, "Error", QString("Loading '%1' failed!").arg(filename));
        return false;
    }
    m_filename[index] = filename;
    if (index == 0) {
        window()->setWindowFilePath(filename);
        window()->setWindowTitle(filename + "[*]");
    }
    m_moduleList->setEnabled(true);
    qDebug() << QString("Loaded #%1:").arg(index) << filename;
    sync();
    return true;
}


void MainWindow::open(int index) {
    QString filename;
    for (int i = index; i >=0; --i) {
        filename = m_filename[i];
        if (!filename.isEmpty()) break;
    }
    filename = QFileDialog::getOpenFileName(this, "Open", filename,
        "Images and Videos (*.png *.bmp *.jpg *.jpeg *.mov *.mp4 *.m4v *.3gp *.avi *.wmv);;All files (*.*)");
    if (!filename.isEmpty()) {
        open(index, filename);
    }
}


void MainWindow::close(int index) {
    m_player[index]->close();
    if (index == 0) {
        m_moduleList->setEnabled(false);
    }
}


void MainWindow::sync() {
    while (m_player[0]->isBusy()) {
        QApplication::processEvents();
    }
    updateCurrentFrame();
}


QString MainWindow::textForModule(const QString& id, Module *module) {
    if (module && m_showFrame) {
        if (id == "title") return module->title();
        if (id == "caption") return module->caption();
        if (id == "title+caption") return module->title() + "\n" + module->caption();
        if (id == "header") return m_globalHeader;
        if (id == "footer") return m_globalFooter;
    }
    return QString::null;
}


void MainWindow::setOutput(Module *module) {
    if (module) {
        /*if (module->input0().isNull()) {
            for (int i = 0; i < 4; ++i) module->setInput(m_player[i]->getFrame(), i);
        }*/
        QImage output = module->output();
        if (m_showFrame) {
//             m_imageView->setImageV( textForModule(m_frameHeader, module),
//                                     output, textForModule(m_frameFooter, module),
//                                     output, textForModule(m_frameFooter, module),
//                                     QSize(m_recWidth,m_recHeight));
            m_imageView->setImage( textForModule(m_frameHeader, module),
                                   output,
                                   textForModule(m_frameFooter, module) );
        } else {
            m_imageView->setImage(output);
        }
    } else {
        m_imageView->setImage(QImage());
    }
}


void MainWindow::on_menuFile_aboutToShow() {
    bool onlyP1 = m_player[0]->isValid() && !m_player[1]->isValid() &&
                 !m_player[2]->isValid()&& !m_player[3]->isValid();

    actionSaveAll->setEnabled(m_moduleList->activeModules().size() > 1);
    actionBatch->setEnabled(m_moduleList->activeModules().size() >= 1);
    actionBatch2->setEnabled(m_moduleList->activeModules().size() >= 2);

    bool a = m_player
        [0]->isValid();
    bool b = (m_player[0]->frameCount() > 1);
    actionRecord->setEnabled(m_player[0]->isValid() && (m_player[0]->frameCount() > 1));
    actionRecord2->setEnabled(m_player[0]->isValid() && (m_player[0]->frameCount() > 1) && (m_moduleList->activeModules().size() >= 2));
}


void MainWindow::on_menuOpen_aboutToShow() {
    actionOpen1->setEnabled(m_player[0]->isValid());
    actionOpen2->setEnabled(m_player[1]->isValid());
    actionOpen3->setEnabled(m_player[2]->isValid());
}


void MainWindow::on_menuClose_aboutToShow() {
    actionClose0->setEnabled(m_player[0]->isValid());
    actionClose1->setEnabled(m_player[1]->isValid());
    actionClose2->setEnabled(m_player[2]->isValid());
    actionClose3->setEnabled(m_player[3]->isValid());
}


void MainWindow::on_actionOpen0_triggered() {
    open(0);
}


void MainWindow::on_actionOpen1_triggered() {
    open(1);
}


void MainWindow::on_actionOpen2_triggered() {
    open(2);
}


void MainWindow::on_actionOpen3_triggered() {
    open(3);
}


void MainWindow::on_actionClose0_triggered() {
    close(0);
}


void MainWindow::on_actionClose1_triggered() {
    close(1);
}


void MainWindow::on_actionClose2_triggered() {
    close(2);
}


void MainWindow::on_actionClose3_triggered() {
    close(3);
}


void MainWindow::updateCurrentFrame() {
    /*
    VideoFrame f[4];
    for (int i = 0; i < 4; ++i) f[i] = m_player[i]->getFrame();

    QList<Module*> L = m_moduleList->modules();
    for (int j = 0; j < L.size(); ++j) {
        for (int i = 0; i < 4; ++i) L[j]->setInput(f[i], i);
    }
    */

    QList<Module*> L = m_moduleList->modules();
    for (int j = 0; j < L.size(); ++j) {
        L[j]->setDirty();
    }
}


void MainWindow::updateModuleOutput() {
    if (m_dirty) {
        Module *m = m_moduleList->current();
        m_imageView->setHandler(m);
        setOutput(m);
        m_dirty = false;
    }
}


void MainWindow::setDirty() {
    if (!m_dirty) {
        m_dirty = true;
        QMetaObject::invokeMethod(this, "updateModuleOutput", Qt::QueuedConnection);
    }
}


void MainWindow::on_actionSavePNG_triggered() {
    m_imageView->savePNG();
}


void MainWindow::on_actionShowInfo_triggered() {
    QSettings settings;
    QString inputPath = window()->windowFilePath();
    QString outputPath = settings.value("savename", inputPath).toString();

    QString filename;
    QFileInfo fi(inputPath);
    QFileInfo fo(outputPath);
    if (!fi.baseName().isEmpty()) {
        QFileInfo fn(fo.dir(), fi.baseName() + "-out.png");
        filename  = fi.absoluteFilePath();
    } else {
        filename  = fo.absolutePath();
    }

    filename = QFileDialog::getOpenFileName(this, "Open PNG", filename,
        "PNG Format (*.png);;All files (*.*)");
    if (!filename.isEmpty()) {
        QImage image(filename);
        if (image.isNull()) {
            QMessageBox::critical(this, "Error", QString("Info PNG '%1' failed!").arg(filename));
            return;
        }
        QString text = image.text("Description");
        text.replace(";", ";\n");
        QMessageBox::information(this, "Show Info", text);
    }
}


void MainWindow::on_actionSavePDF_triggered() {
    Module *module = m_moduleList->current();
    QString text = QString("%1; %2").arg(module->objectName()).arg(AbstractParam::paramText(module));
    m_imageView->savePDF(text);
}


void MainWindow::on_actionSaveAll_triggered() {
    QList<Module*> ML = m_moduleList->activeModules();
    if (ML.count() < 1)
        return;

    QSettings settings;
    QString filename;
    {
        QFileInfo fi(m_filename[0]);
        QFileInfo fo(settings.value("savename", m_filename[0]).toString());
        filename = QFileDialog::getSaveFileName(this, "Save All", fo.path() + "/" + fi.baseName() + ".png",
            "PNG Format (*.png);;JPG Format (*.jpg);;All files (*.*)");
    }
    if (!filename.isEmpty()) {
        setEnabled(false);
        QProgressDialog progress("Batch Processing...", "Abort", 0, ML.count(), this);
        progress.setWindowModality(Qt::WindowModal);

        QFileInfo fi(filename);
        for (int k = 0; k < ML.size(); ++k) {
            QString fn = fi.absolutePath() + QString("/%1-%2.%3").arg(fi.baseName()).arg(k+1).arg(fi.suffix());
            progress.setValue(k);
            progress.setLabelText(fn);

            setOutput(ML[k]);
            if (!m_imageView->image().save(fn, 0, 95)) {
                progress.cancel();
                QMessageBox::critical(this, "Error", QString("Saving image '%1' failed!").arg(fn));
                break;
            }

            if (progress.wasCanceled()) break;
        }
        setEnabled(true);
        settings.setValue("savename", filename);
        setOutput(m_moduleList->current());
    }
}


void MainWindow::on_actionBatch_triggered() {
    QList<Module*> ML = m_moduleList->activeModules();
    if (ML.count() < 1)
        return;

    QString batchInput = QFileDialog::getExistingDirectory(this, "Choose input directory", m_filename[0]);
    if (batchInput.isEmpty())
        return;

    QSettings settings;
    QString batchOutput = QFileDialog::getExistingDirectory(this, "Choose output directory",
        settings.value("savename", m_filename[0]).toString());
    if (batchOutput.isEmpty())
        return;

    QDir dir(batchInput);
    dir.setFilter(QDir::Files);
    QFileInfoList list = dir.entryInfoList();

    setEnabled(false);
    QProgressDialog progress("Batch Processing...", "Abort", 0, ML.size() * list.size(), this);
    progress.setWindowModality(Qt::WindowModal);

    for (int i = 0; i < list.size(); ++i) {
        QFileInfo fi = list[i];

        if (!open(0, fi.absoluteFilePath())) {
            QMessageBox::critical(this, "Error", QString("Loading image '%1' failed!").arg(fi.absoluteFilePath()));
            break;
        }

        for (int k = 0; k < ML.size(); ++k) {
            QString fn;
            if (ML.size() == 1) {
                fn = batchOutput + QString("/%1.%2").arg(fi.baseName()).arg(m_batchFormat);
            } else {
                fn = batchOutput + QString("/%1-%2.%3").arg(fi.baseName()).arg(k+1).arg(m_batchFormat);
            }
            progress.setValue(i * ML.size() + k);
            progress.setLabelText(fn);

            setOutput(ML[k]);

            if (!m_imageView->image().save(fn, 0, 95)) {
                QMessageBox::critical(this, "Error", QString("Saving image '%1' failed!").arg(fn));
                progress.cancel();
                break;
            }

            qApp->processEvents();
            if (progress.wasCanceled())
                break;
        }
        if (progress.wasCanceled())
            break;

        update();
        qApp->processEvents();

    }

    settings.setValue("savename", batchOutput);
    setEnabled(true);
    setOutput(m_moduleList->current());
}


void MainWindow::on_actionBatchEx_triggered() {
    QList<Module*> ML = m_moduleList->activeModules();
    if (ML.count() < 1)
        return;

    QString batchInput = QFileDialog::getExistingDirectory(this, "Choose input directory", m_filename[0]);
    if (batchInput.isEmpty())
        return;

    QSettings settings;
    QString batchOutput = QFileDialog::getExistingDirectory(this, "Choose output directory",
        settings.value("savename", m_filename[0]).toString());
    if (batchOutput.isEmpty())
        return;

    QDir dir(batchInput);
    dir.setFilter(QDir::Files);
    QFileInfoList list = dir.entryInfoList();

    setEnabled(false);
    QProgressDialog progress("Batch Processing...", "Abort", 0, ML.size() * list.size(), this);
    progress.setWindowModality(Qt::WindowModal);

    QString outname;
    {
        QFileInfo fi(batchOutput);
        outname = fi.baseName();
    }

    int fileCount = 0;
    for (int i = 0; i < list.size(); ++i) {
        QFileInfo fi = list[i];
        if (!fi.baseName().endsWith("-0"))
            continue;

        if (!open(0, fi.absoluteFilePath())) {
            QMessageBox::critical(this, "Error", QString("Loading image '%1' failed!").arg(fi.absoluteFilePath()));
            break;
        }

        //QImage input(fi.absoluteFilePath());
        //if (input.isNull()) {
        //VideoFrame inputFrame(input);

        QString outname2;
        {
            int a = fileCount % 26;
            int b = fileCount / 26;
            outname2 = outname + "_" + QString('a' + b) + QString('a' + a);
        }

        for (int k = 0; k < ML.size(); ++k) {
            QString fn;
            if (ML.size() == 1) {
                fn = batchOutput + QString("/%1.%3").arg(outname2).arg(m_batchFormat);
            } else {
                fn = batchOutput + QString("/%1-%2.%3").arg(outname2).arg(k+1).arg(m_batchFormat);
            }
            progress.setValue(i * ML.size() + k);
            progress.setLabelText(fn);

            //ML[k]->setInput(inputFrame, 0);
            setOutput(ML[k]);

            if (!m_imageView->image().save(fn, 0, 95)) {
                QMessageBox::critical(this, "Error", QString("Saving image '%1' failed!").arg(fn));
                progress.cancel();
                break;
            }

            qApp->processEvents();
            if (progress.wasCanceled())
                break;
        }
        if (progress.wasCanceled())
            break;

        ++fileCount;
        update();
        qApp->processEvents();

    }

    {
        QSettings infoFile(batchOutput + "/settings.ini", QSettings::IniFormat);
        for (int k = 0; k < ML.size(); ++k) {
            infoFile.beginGroup(ML[k]->objectName());
            AbstractParam::saveSettings(infoFile, ML[k]);
            infoFile.endGroup();
        }
    }

    settings.setValue("savename", batchOutput);
    setEnabled(true);
    setOutput(m_moduleList->current());
}


void MainWindow::on_actionBatch2_triggered() {
    QList<Module*> ML = m_moduleList->activeModules();
    Module* current = m_moduleList->current();
    ML.removeOne(current);
    if (ML.size() < 1)
        return;

    QString batchInput = QFileDialog::getExistingDirectory(this, "Choose input directory", m_filename[0]);
    if (batchInput.isEmpty())
        return;

    QSettings settings;
    QString batchOutput = QFileDialog::getExistingDirectory(this, "Choose output directory",
        settings.value("savename", m_filename[0]).toString());
    if (batchOutput.isEmpty())
        return;

    QDir dir(batchInput);
    dir.setFilter(QDir::Files);
    QFileInfoList list = dir.entryInfoList();

    setEnabled(false);
    QProgressDialog progress("Batch Processing...", "Abort", 0, ML.count() * list.size(), this);
    progress.setWindowModality(Qt::WindowModal);

    for (int i = 0; i < list.size(); ++i) {
        QFileInfo fi = list[i];

        if (!open(0, fi.absoluteFilePath())) {
            QMessageBox::critical(this, "Error", QString("Loading image '%1' failed!").arg(fi.absoluteFilePath()));
            break;
        }
        QImage currentOutput = current->output();
        QString currentFooter = textForModule(m_frameFooter, current);

        for (int k = 0; k < ML.count(); ++k) {
            QString fn;
            if (ML.size() ==  1)
                fn = batchOutput + QString("/%1.%2").arg(fi.baseName()).arg(m_batchFormat);
            else
                fn = batchOutput + QString("/%1-%2.%3").arg(fi.baseName()).arg(k+1).arg(m_batchFormat);
            progress.setValue(i * ML.size() + k);
            progress.setLabelText(fn);

            Module *m = ML[k];
            QImage output = m->output();
            m_imageView->setImageH( textForModule(m_frameHeader, current),
                                    output,
                                    textForModule(m_frameFooter, m),
                                    currentOutput,
                                    currentFooter );

            if (!m_imageView->image().save(fn, 0, 95)) {
                QMessageBox::critical(this, "Error", QString("Saving image '%1' failed!").arg(fn));
                progress.cancel();
                break;
            }

            qApp->processEvents();
            if (progress.wasCanceled())
                break;
        }
        if (progress.wasCanceled())
            break;

        update();
        qApp->processEvents();

    }

    settings.setValue("savename", batchOutput);
    setEnabled(true);
    setOutput(m_moduleList->current());
}


void MainWindow::on_actionRecord_triggered() {
    Module *m = m_moduleList->current();

    QSettings settings;
    QFileInfo fi(m_filename[0]);
    QFileInfo fo(settings.value("savename", m_filename[0]).toString());
    QString filename = QFileDialog::getSaveFileName(this, "Record", fo.dir().filePath(fi.baseName() + "-out"),
        "RGB lossless HuffYUV (*.avi);;MPEG 4 (*.mp4);;All files (*.*)");
    if (filename.isEmpty())
        return;

    unsigned w = m_imageView->image().width();
    unsigned h = m_imageView->image().height();
    if (!m_recAutoSize) {
        w = m_recWidth;
        h = m_recHeight;
    }
    unsigned nframes = m_player[0]->frameCount();

    QPair<int,int> frameRate = m_player[0]->frameRate();
    if (m_recAutoFps || (m_player[0]->fps() > 30))
        frameRate = QPair<int,int>(m_recFps*1000, 1000);

    oz::ffmpeg_encoder *encoder = oz::ffmpeg_encoder::create(filename.toStdString().c_str(), w, h,
                                   std::pair<int,int>(frameRate.first, frameRate.second), m_recBitRate);
    if (!encoder) {
        QMessageBox::critical(this, "Error", QString("Creation of %1 failed!").arg(filename));
        return;
    }

    setEnabled(false);
    QProgressDialog progress("Recording...", "Abort", 0, nframes-1, this);
    progress.setWindowModality(Qt::WindowModal);

    for (unsigned i = 0; i < nframes; ++i) {
        progress.setValue(i);

        m_player[0]->setCurrentFrame(i);
        sync();
        setOutput(m);
        QImage image = m_imageView->image();
        QImage e(w, h, QImage::Format_RGB32);
        e.fill(0);
        QPainter p(&e);
        p.drawImage(w/2-m_imageView->image().width()/2, h/2-m_imageView->image().height()/2, image);
        encoder->append_frame(e.bits());

        qApp->processEvents();
        if (progress.wasCanceled())
            break;
    }

    encoder->finish();
    delete encoder;

    settings.setValue("savename", filename);
    setEnabled(true);
    QDesktopServices::openUrl(QUrl::fromLocalFile(filename));
}


void MainWindow::on_actionRecord2_triggered() {
    if (!m_player)
        return;
    QList<Module*> ML = m_moduleList->activeModules();
    if (ML.size() < 1)
        return;
    ML.removeOne(m_moduleList->current());
    ML.prepend(m_moduleList->current());

    unsigned w = m_imageView->image().width();
    unsigned h = m_imageView->image().height();
    if (m_recOrient == "horizontal") {
        w = w*2 + 30;
        h = h + 70;
    } else {
        w = w + 20;
        h = 2 * h + 30 + 100;
    }
    if (!m_recAutoSize) {
        w = m_recWidth;
        h = m_recHeight;
    }

    QSettings settings;
    QFileInfo fi(m_filename[0]);
    QFileInfo fo(settings.value("savename", m_filename[0]).toString());
    QString filename = QFileDialog::getSaveFileName(this, "Record", fo.dir().filePath(fi.baseName() + "-out"),
        "Uncompressed AVI (*.avi);;MPEG 4 (*.mp4);;All files (*.*)");
    if (filename.isEmpty())
        return;

    unsigned nframes = m_player[0]->frameCount();

    QPair<int,int> frameRate = m_player[0]->frameRate();
    if (m_recAutoFps || (m_player[0]->fps() > 30))
        frameRate = QPair<int,int>(m_recFps*1000, 1000);

    oz::ffmpeg_encoder *encoder = oz::ffmpeg_encoder::create(filename.toStdString().c_str(), w, h,
                                   std::pair<int,int>(frameRate.first, frameRate.second), m_recBitRate);
    if (!encoder) {
        QMessageBox::critical(this, "Error", QString("Creation of %1 failed!").arg(filename));
        return;
    }

    setEnabled(false);
    QProgressDialog progress("Recording...", "Abort", 0, (ML.count()-1) * (nframes-1), this);
    progress.setWindowModality(Qt::WindowModal);

    if (!m_titleImage.isNull()) {
        QImage img(w, h, QImage::Format_RGB32);
        QPainter p(&img);
        img.fill(0);
        p.drawImage(w/2 -  m_titleImage.width()/2, h/2-m_titleImage.height()/2,  m_titleImage);
        for (int k = 0; k < 2*m_player[0]->fps(); ++k) {
            encoder->append_frame(img.bits());
        }
    }
    for (int k = 1; k < ML.size(); ++k) {
        for (unsigned i = 0; i < nframes; ++i) {
            progress.setValue((k-1) * nframes + i);

            m_player[0]->setCurrentFrame(i);
            sync();

            QImage image0 = ML[0]->output();
            QImage imageK = ML[k]->output();
            if (m_recOrient == "horizontal") {
                m_imageView->setImageH( textForModule(m_frameHeader, ML[0]),
                                        imageK, textForModule(m_frameFooter, ML[k]),
                                        image0, textForModule(m_frameFooter, ML[0]),
                                        QSize(w,h));
            } else {
                m_imageView->setImageV( textForModule(m_frameHeader, ML[0]),
                                        imageK, textForModule(m_frameFooter, ML[k]),
                                        image0, textForModule(m_frameFooter, ML[0]),
                                        QSize(w,h));
            }

            QImage image = m_imageView->image();
            encoder->append_frame(image.bits());

            qApp->processEvents();
            if (progress.wasCanceled())
                break;
        }
        if (progress.wasCanceled())
            break;
    }
    if (!m_creditsImage.isNull()) {
        QImage img(w, h, QImage::Format_RGB32);
        QPainter p(&img);
        img.fill(0);
        p.drawImage(w/2 - m_creditsImage.width()/2, h/2-m_creditsImage.height()/2,  m_creditsImage);
        for (int k = 0; k < 2*m_player[0]->fps(); ++k) {
            encoder->append_frame(img.bits());
        }
    }
    encoder->finish();

    settings.setValue("savename", filename);
    setEnabled(true);
    QDesktopServices::openUrl(QUrl::fromLocalFile(filename));
}


void MainWindow::on_actionModules_triggered() {
    ModuleList::edit(this, m_moduleList);
}


void MainWindow::on_actionAbout_triggered() {
    QMessageBox msgBox;
    msgBox.setWindowTitle("About");
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setText(
        "<html><body>" \
        "<b>Image and Video Abstraction Test Application</b><br/><br/>" \
        "Copyright (C) 2011 Hasso-Plattner-Institut,<br/>" \
        "Fachgebiet Computergrafische Systeme &lt;" \
        "<a href='http://www.hpi3d.de'>www.hpi3d.de</a>&gt;<br/><br/>" \
        "Author: Jan Eric Kyprianidis &lt;" \
        "<a href='http://www.kyprianidis.com'>www.kyprianidis.com</a>&gt;<br/>" \
        "Date: " __DATE__ "<br/>" \
        "</body></html>"
    );
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}


void MainWindow::on_actionSelectDevice_triggered() {
    int current = 0;
    cudaGetDevice(&current);
    int N = CudaDeviceDialog::select(true);
    if ((N >= 0) && (current != N)) {
        QMessageBox::information(this, "Information", "Application must be restarted!");
        qApp->quit();
    }
}