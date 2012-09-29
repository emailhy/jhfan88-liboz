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
#include "simplemodule.h"
#include "paramui.h"
#include <oz/log.h>


SimpleModule::SimpleModule() : Module() {
}


SimpleModule::~SimpleModule() {
}


void SimpleModule::saveSettings(QSettings& settings) {
    Module::saveSettings(settings);
    settings.setValue("currentKey", m_currentKey);

}


void SimpleModule::restoreSettings(QSettings& settings) {
    Module::restoreSettings(settings);
    m_currentKey = settings.value("currentKey").toString();
}


QWidget* SimpleModule::createUI() {
    QWidget *w = new QWidget;
    QVBoxLayout *vbox = new QVBoxLayout(w);
    vbox->setContentsMargins(4,4,4,4);
    vbox->setSpacing(4);

    ModuleComboBox *cb = new ModuleComboBox(w, this);
    vbox->addWidget(cb);

    ParamUI *ui = new ParamUI(w, this);
    vbox->addWidget(ui);
    vbox->addStretch(0);

    const QMetaObject* meta = metaObject();
    if (meta->indexOfSlot("benchmark()") >= 0) {
        QPushButton *b = new QPushButton("Benchmark", w);
        connect(b, SIGNAL(clicked()), this, SLOT(benchmark()));
        vbox->addWidget(b);
    }

    if (meta->indexOfSlot("test()") >= 0) {
        QPushButton *b = new QPushButton("Test", w);
        connect(b, SIGNAL(clicked()), this, SLOT(test()));
        vbox->addWidget(b);
    }

    return w;
}


static void log_callback(const oz::cpu_image& image, const char* msg, void *user) {
    SimpleModule *self = (SimpleModule*)user;
    QString k =  msg;
    QStringList keys = self->publishedImageKeys();
    if (keys.contains(k)) {
        for (int no = 2;; ++no) {
            k = QString("%1%2").arg(msg).arg(no);
            if (!keys.contains(k)) break;
        }
    }
    self->publish(k, image);
}


void SimpleModule::check() {
    if (!isDirty()) return;
    clearPublishedImages();
    if (!player()->image().isNull()) {
        qDebug() << "Processing" << objectName();
        try {
            oz::install_log_handler(log_callback, this);
            process();
            oz::install_log_handler(NULL, NULL);
        }
        catch ( std::exception& e ) {
            qWarning() << e.what();
        }
    } else {
        setOutput(QImage());
    }
    if (m_currentKey.isEmpty() && !m_images.isEmpty()) {
        m_currentKey = m_images.keys().first();
    }
    setOutput(getImage(m_currentKey));
    finishedProcessing();
}


void SimpleModule::clearPublishedImages() {
    m_images.clear();
}


void SimpleModule::publish(const QString& key, const oz::cpu_image& img ) {
    m_images.take(key);
    if (img.is_valid()) {
        m_images.insert(key, img);
    }
}


void SimpleModule::publish(const QString& key, const QImage& image) {
    publish(key, oz::from_qimage(image));
}


void SimpleModule::publishHistogram(const QString& key, const std::vector<int>& H) {
    int N = (int)H.size();
    double m, s;
    {
        double m1 = 0;
        double m2 = 0;
        for (int i = 0; i < 256; ++i) {
            m1 += H[i];
            m2 += H[i]*H[i];
        }
        m1 /= 256;
        m2 /= 256;
        m = m1;
        s = sqrt(abs(m2 - m1*m1));
    }

    int M;
    if (0/*scale*/) {
        M = 3*s;
    } else {
        M = 0;
        for (int i = 0; i < N; ++i) {
            if (M < H[i]) M = H[i];
        }
    }

    QImage I(N+2, 256+2, QImage::Format_RGB32);
    QPainter p(&I);
    p.fillRect(I.rect(), Qt::lightGray);
    if (M > 0) {
        for (int i = 0; i < N; ++i) {
            int y = 256 * H[i] / M;
            if (y > 0) p.fillRect(i+1, 1+256-y, 1, y, Qt::darkGray);
        }
    }
    /*
    if (ticks) {
        for (int i = 0; i <= ticks; ++i) {
            int x = i*N/ticks;
            p.drawLine(x, 256, x, 256+2);
        }
    }
    */
    publish(key, I);
}


void SimpleModule::publishHistogram(const QString& key, const std::vector<int>& H, float M) {
    int N = (int)H.size();
    QImage I(N+2, 256, QImage::Format_RGB32);
    QPainter p(&I);
    p.fillRect(I.rect(), Qt::lightGray);
    if (M > 0) {
        for (int i = 0; i < N; ++i) {
            int y = 256 * H[i] / M;
            if (y > 0) p.fillRect(i+1, 256-y, 1, y, Qt::darkGray);
        }
    }
    publish(key, I);
}


void SimpleModule::publish(const QString& key, const std::vector<float>& pdf) {
    int N = (int)pdf.size();
    QImage I(N+2, 256, QImage::Format_RGB32);
    QPainter p(&I);
    p.fillRect(I.rect(), Qt::lightGray);
    float M = 0;
    std::vector<float> s = pdf;
    sort(s.begin(), s.end());
    M = s[98 * N / 100] * 100.0f / 98.0f;
    if (M > 0) {
        for (int i = 0; i < N; ++i) {
            int y = 256 * pdf[i] / M;
            if (y > 0) p.fillRect(i+1, 256-y, 1, y, Qt::darkGray);
        }
    }
    publish(key, I);
}


void SimpleModule::publishVector(const QString& key, const std::vector<oz::gpu_image>& P, int spacing) {
    int w = 0;
    int h = 0;
    for (unsigned i = 0; i < P.size(); ++i) {
        w += P[i].w();
        h = std::max<int>(h, P[i].h());
    }
    double delta;
    if (spacing >= 0) {
        w += spacing * ((int)P.size() - 1);
        delta = spacing;
    } else {
        delta = (-spacing - w) / ((int)P.size() - 1);
        w = -spacing;
    }

    QImage img(w, h, QImage::Format_RGB32);
    img.fill(0xffffffff);
    //img.fill(0xffc0c0c0);
    QPainter p(&img);
    double x = 0;
    for (unsigned i = 0; i < P.size(); ++i) {
        QImage I = to_qimage(P[i]);
        p.drawImage((int)x, (h-I.height())/2, I);
        x += I.width() + delta;
    }
    publish(key, img);
}


void SimpleModule::publishPyr(const QString& key, const std::vector<oz::gpu_image>& P, int orientation) {
    if (orientation < 0) {
        orientation = P[0].w() < P[0].h();
    }

    int w, h;
    if (orientation == 0) {
        w = P[0].w() + P[1].w() + 10;
        h = P[0].h();
    } else {
        w = P[0].w();
        h = P[0].h() + P[1].h() + 10;
    }

    QImage img(w, h, QImage::Format_RGB32);
    img.fill(0xffffffff);
    QPainter p(&img);

    int x = 0;
    int y = 0;
    for (unsigned i = 0; i < P.size(); ++i) {

        QImage I = to_qimage(P[i]);
        p.drawImage(x, y, I);

        if ((i + orientation) & 1) {
            y += P[i].h() + 10;
        } else {
            x += P[i].w() + 10;
        }
    }

    publish(key, img);
}


const QStringList SimpleModule::publishedImageKeys() const {
    return m_images.keys();
}


oz::cpu_image SimpleModule::publishedImage(const QString& key) const {
    return m_images.value(key);
}

QImage SimpleModule::getImage(const QString& key) const {
    return oz::to_qimage(publishedImage(key));
}


void SimpleModule::setCurrentKey(const QString& key) {
    if (m_currentKey != key) {
        m_currentKey = key;
        setOutput(getImage(key));
        currentKeyChanged(key);
    }
}


ModuleComboBox::ModuleComboBox(QWidget *parent, SimpleModule *module)
    : QComboBox(parent), m_module(module)
{
    connect(this, SIGNAL(currentIndexChanged(const QString&)), module, SLOT(setCurrentKey(const QString&)));
    connect(module, SIGNAL(currentKeyChanged(const QString&)), this, SLOT(setCurrentKey(const QString&)));
    connect(module, SIGNAL(finishedProcessing()), this, SLOT(updatePublishedImages()));
}


void ModuleComboBox::setCurrentKey(const QString& key) {
    QStringList K = m_module->publishedImageKeys();
    setCurrentIndex(K.indexOf(key));
}


void ModuleComboBox::updatePublishedImages() {
    bool sb = signalsBlocked();
    blockSignals(true);
    QStringList K = m_module->publishedImageKeys();
    clear();
    addItems(K);
    QString key = m_module->currentKey();
    setCurrentIndex(K.indexOf(key));
    blockSignals(sb);
}

