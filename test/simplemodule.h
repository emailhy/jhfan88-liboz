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
#include <oz/qimage.h>

class SimpleModule : public Module {
    Q_OBJECT
public:
    SimpleModule();
    ~SimpleModule();

    virtual void saveSettings(QSettings& settings);
    virtual void restoreSettings(QSettings& settings);
    virtual QWidget* createUI();

    void clearPublishedImages();
    void publish(const QString& key, const oz::cpu_image& img);
    void publish(const QString& key, const QImage& img);
    void publish(const QString& key, const std::vector<float>& pdf);
    void publishHistogram(const QString& key, const std::vector<int>& H);
    void publishHistogram(const QString& key, const std::vector<int>& H, float M);
    void publishVector(const QString& key, const std::vector<oz::gpu_image>& P, int spacing=0);
    void publishPyr(const QString& key, const std::vector<oz::gpu_image>& P, int orientation=-1);

    const QStringList publishedImageKeys() const;
    QString currentKey() const { return m_currentKey; }
    oz::cpu_image publishedImage(const QString& key) const;
    QImage getImage(const QString& key) const;

    oz::cpu_image cpuInput0(int index=0) const {
        return oz::from_qimage(this->player()->image(index));
    }

    oz::gpu_image gpuInput0(int index=0) const {
        oz::gpu_image img = oz::from_qimage(this->player(0)->image(index));
        return img.convert(oz::FMT_FLOAT3);
    }
    oz::gpu_image gpuInput1(int index=0) const {
        oz::gpu_image img = oz::from_qimage(this->player(1)->image(index));
        return img.convert(oz::FMT_FLOAT3);
    }
    oz::gpu_image gpuInput2(int index=0) const {
        oz::gpu_image img = oz::from_qimage(this->player(2)->image(index));
        return img.convert(oz::FMT_FLOAT3);
    }
    oz::gpu_image gpuInput3(int index=0) const {
        oz::gpu_image img = oz::from_qimage(this->player(3)->image(index));
        return img.convert(oz::FMT_FLOAT3);
    }

public slots:
    void setCurrentKey(const QString&);

signals:
    void finishedProcessing();
    void currentKeyChanged(const QString&);

protected:
    virtual void check();

    QString m_currentKey;
    QMap<QString,oz::cpu_image> m_images;
};


class ModuleComboBox : public QComboBox {
    Q_OBJECT
public:
    ModuleComboBox(QWidget *parent, SimpleModule *module);

protected slots:
    void setCurrentKey(const QString& key);
    void updatePublishedImages();

protected:
    SimpleModule *m_module;
};
