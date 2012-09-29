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

#include "imageview.h"

class ImageView2 : public ImageView {
    Q_OBJECT
    Q_ENUMS(Layout)
    Q_PROPERTY(bool printPreview READ printPreview WRITE setPrintPreview NOTIFY printPreviewChanged)
    Q_PROPERTY(int paperWidth READ paperWidth WRITE setPaperWidth NOTIFY paperWidthChanged)
    Q_PROPERTY(int paperHeight READ paperHeight WRITE setPaperHeight NOTIFY paperHeightChanged)
    Q_PROPERTY(bool showGrid READ showGrid WRITE setShowGrid NOTIFY showGridChanged)
    Q_PROPERTY(bool printFrame READ printFrame WRITE setPrintFrame NOTIFY printFrameChanged)
    Q_PROPERTY(bool colorCorrect READ colorCorrect WRITE setColorCorrect NOTIFY colorCorrectChanged)
    Q_PROPERTY(double saturation READ saturation WRITE setSaturation NOTIFY saturationChanged)
    Q_PROPERTY(double brightness READ brightness WRITE setBrightness NOTIFY brightnessChanged)
    Q_PROPERTY(double contrast READ contrast WRITE setContrast NOTIFY contrastChanged)
    Q_PROPERTY(Layout layout READ layout WRITE setLayout NOTIFY layoutChanged)
    Q_PROPERTY(int gap READ gap WRITE setGap NOTIFY gapChanged)
    Q_PROPERTY(double frameWidth READ frameWidth WRITE setFrameWidth NOTIFY frameWidthChanged)
    Q_PROPERTY(double lineWidth READ lineWidth WRITE setLineWidth NOTIFY lineWidthChanged)
public:
    ImageView2(QWidget *parent);
    ~ImageView2();

    virtual void restoreSettings(QSettings& settings);
    virtual void saveSettings(QSettings& settings);

    enum Layout {
        None,
        Left_2X,
        Right_2X,
        Both_3X,
        Top_2X,
        Bottom_2X
    };

    bool printPreview() const { return m_printPreview; }
    int paperWidth() const { return m_paperSize.width(); }
    int paperHeight() const { return m_paperSize.height(); }
    bool showGrid() const { return m_showGrid; }
    bool printFrame() const { return m_printFrame; }
    bool colorCorrect() const { return m_colorCorrect; }
    double saturation() const { return m_saturation; }
    double brightness() const { return m_brightness; }
    double contrast() const { return m_contrast; }
    Layout layout() const { return m_layout; }
    int gap() const { return m_gap; }
    double frameWidth() const { return m_frameWidth; }
    double lineWidth() const { return m_lineWidth; }

    virtual float pt2px(float pt) const;

public slots:
    void setImage(const QImage& image);
    void setImage(const QString& title, const QImage& image, const QString& caption);
    void setImageH(const QString& title, const QImage& image1, const QString& caption1,
                   const QImage& image2, const QString& caption2, const QSize& size = QSize());
    void setImageV(const QString& title, const QImage& image1, const QString& caption1,
                   const QImage& image2, const QString& caption2, const QSize& size = QSize());
    void setPrintPreview(bool flag);
    void setPaperWidth(int value);
    void setPaperHeight(int value);
    void setShowGrid(bool value);
    void setPrintFrame(bool value);
    void setColorCorrect(bool value);
    void setSaturation(double value);
    void setBrightness(double value);
    void setContrast(double value);
    void setLayout(Layout value);
    void setGap(int value);
    void setFrameWidth(double value);
    void setLineWidth(double value);
    void fitWidth();
    void fitHeight();
    void savePDF(const QString& text=QString());

signals:
    void printPreviewChanged(bool);
    void paperWidthChanged(int);
    void paperHeightChanged(int);
    void showGridChanged(bool);
    void printFrameChanged(bool);
    void colorCorrectChanged(bool);
    void saturationChanged(double);
    void brightnessChanged(double);
    void contrastChanged(double);
    void layoutChanged(Layout);
    void gapChanged(int);
    void frameWidthChanged(double);
    void lineWidthChanged(double);

protected:
    virtual void paintEvent( QPaintEvent* );
    virtual void paint(QPainter &p, const QSizeF& sz);
    virtual void paint(QPainter &p, const QSizeF& sz, double zoom, const QPointF& origin, int pass);
    virtual void draw(QPainter& p, const QImage& image, int pass);

protected:
    bool m_printPreview;
    bool m_printing;
    QSize m_paperSize;
    bool m_showGrid;
    bool m_printFrame;
    bool m_colorCorrect;
    double m_saturation;
    double m_brightness;
    double m_contrast;
    Layout m_layout;
    int m_gap;
    double m_frameWidth;
    double m_lineWidth;
};
