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
#include "imageview2.h"


ImageView2::ImageView2(QWidget *parent) : ImageView(parent) {
    m_printing = false;
    m_printPreview = false;
    m_paperSize = QSize(320, 240);
    m_showGrid = false;
    m_printFrame = false;
    m_colorCorrect = false;
    m_saturation = 0;
    m_brightness = 0.66;
    m_contrast = 0.25;
    m_layout = None;
    m_gap = 4;
    m_frameWidth = 0.25;
    m_lineWidth = 1;
}


ImageView2::~ImageView2() {
}


void ImageView2::restoreSettings(QSettings& settings) {
    ImageView::restoreSettings(settings);
    setPrintPreview(settings.value("printPreview", false).toBool());
    setPaperWidth(settings.value("paperWidth", 320).toInt());
    setPaperHeight(settings.value("paperHeight", 240).toInt());
    setShowGrid(settings.value("showGrid", false).toBool());
    setPrintFrame(settings.value("printFrame", false).toBool());
    setColorCorrect(settings.value("colorCorrect", false).toBool());
    setSaturation(settings.value("saturation", 1).toDouble());
    setBrightness(settings.value("brightness", 0).toDouble());
    setContrast(settings.value("contrast", 1).toDouble());
    setLayout((Layout)settings.value("layout", 0).toInt());
    setGap(settings.value("gap", 4).toInt());
    setFrameWidth(settings.value("frameWidth", 0.25).toDouble());
    setLineWidth(settings.value("lineWidth", 1).toDouble());
    update();
}


void ImageView2::saveSettings(QSettings& settings) {
    ImageView::saveSettings(settings);
    settings.setValue("printPreview", m_printPreview);
    settings.setValue("paperWidth", m_paperSize.width());
    settings.setValue("paperHeight", m_paperSize.height());
    settings.setValue("showGrid", m_showGrid);
    settings.setValue("printFrame", m_printFrame);
    settings.setValue("colorCorrect", m_colorCorrect);
    settings.setValue("saturation", m_saturation);
    settings.setValue("brightness", m_brightness);
    settings.setValue("contrast", m_contrast);
    settings.setValue("layout", m_layout);
    settings.setValue("gap", m_gap);
    settings.setValue("frameWidth", m_frameWidth);
    settings.setValue("lineWidth", m_lineWidth);
}


float ImageView2::pt2px(float pt) const {
    float px = pt;
    if (m_printing && (px < 0.25f)) px = 0.25f;
    px /= zoom();
    return px;
}


void ImageView2::setImage(const QImage& image) {
    ImageView::setImage(image);
}


void ImageView2::setImage(const QString& title, const QImage& image, const QString& caption) {
    if (image.isNull()) {
        setImage(image);
        return;
    }

    int w = image.width() + 20;
    int h = image.height() + 20;
    if (!title.isEmpty()) h += 50;
    if (!caption.isEmpty()) h += 50;

    QColor fg(Qt::black);
    QColor bg(Qt::white);
    /*{
        int sum = 0;
        for (int i = 0; i < image.width(); ++i) {
            QRgb c = image.pixel(i,0);
            int r = qRed(c);
            int g = qGreen(c);
            int b = qBlue(c);
            sum += sqrtf(r*r + g*g + b*b) / sqrtf(3.0f);
        }
        sum /= image.width();
        if (sum > 240) {
            bg = QColor(Qt::black);
            fg = QColor(Qt::white);
        }
    }*/

    QImage img(w, h, QImage::Format_RGB32);
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);
    p.setPen(QPen(fg));

    img.fill(bg.rgb());
    int x = 10;
    int y = 10;

    if (!title.isEmpty()) {
        p.setFont(QFont("Lucida Sans Unicode", 12, QFont::Normal));
        p.drawText(QRect(0, y, w, 40), Qt::AlignCenter, title);
        y += 50;
    }

    p.drawImage(x, y, image);
    y += image.height();

    if (!caption.isEmpty()) {
        QRect R(0, y, w, 60);
        QFont f("Lucida Sans Unicode", 10, QFont::Normal);
        for (float pt = 10; pt >= 5; pt -= 0.25f) {
            f.setPointSizeF(pt);
            QFontMetrics m(f, &img);
            QRect br = m.boundingRect(R, Qt::AlignCenter, caption);
            if ((br.width() <= w-10) && (br.height() <= 45)) break;
        }
        p.setFont(f);
        p.drawText(R, Qt::AlignCenter, caption);
    }

    setImage(img);
}


void ImageView2::setImageH(const QString& title, const QImage& image1, const QString& caption1,
                                                const QImage& image2, const QString& caption2,
                                                const QSize& size) {
    int ph = 20;
    if (!title.isEmpty()) ph += 50;
    if (!caption1.isEmpty() || !caption2.isEmpty()) ph += 50;
    int w, h;
    if (size.isValid()) {
        w = size.width();
        h = size.height();
    } else {
        w = 2 * qMax(image1.width(), image2.width()) + 30;
        h = qMax(image1.height(), image2.height()) + ph;
    }

    int cw = (w - 30) / 2;
    int ch = qMin(h - ph, qMax(image1.height(), image2.height()));

    QImage img(w, h, QImage::Format_RGB32);
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);
    p.setPen(QPen(Qt::black));
    img.fill(0xffffffff);
    int x = 10;
    int y = 10 + (h - ch - ph) / 2;

    if (!title.isEmpty()) {
        p.setFont(QFont("Lucida Sans Unicode", 12, QFont::Normal));
        p.drawText(QRect(0, y, w, 40), Qt::AlignCenter, title);
        y += 50;
    }

    {
        int iw = qMin(cw, image1.width());
        int ih = qMin(ch, image1.height());
        int sx = (image1.width() - iw) / 2;
        int sy = (image1.height() - ih) / 2;
        int dx = qMax(0, (cw - iw) / 2);
        int dy = qMax(0, (ch - ih) / 2);
        p.drawImage(x+dx, y+dy, image1.copy(sx, sy, iw, ih));
    }
    {
        int iw = qMin(cw, image2.width());
        int ih = qMin(ch, image2.height());
        int sx = (image2.width() - iw) / 2;
        int sy = (image2.height() - ih) / 2;
        int dx = qMax(0, (cw - iw) / 2);
        int dy = qMax(0, (ch - ih) / 2);
        p.drawImage(5+w/2+dx, y+dy, image2.copy(sx, sy, iw, ih));
    }
    y += ch + 5;

    if (!caption1.isEmpty()) {
        QRect R(0, y, w/2, 50);
        QFont f("Lucida Sans Unicode", 10, QFont::Normal);
        for (float pt = 10; pt >= 5; pt -= 0.25f) {
            f.setPointSizeF(pt);
            QFontMetrics m(f, &img);
            QRect br = m.boundingRect(R, Qt::AlignCenter, caption1);
            if ((br.width() <= w/2) && (br.height() <= 45)) break;
        }
        p.setFont(f);
        p.drawText(R, Qt::AlignCenter, caption1);
        //p.setFont(QFont("Lucida Sans Unicode", 10, QFont::Normal));
        //p.drawText(QRect(0, y, w/2, 40), Qt::AlignCenter, caption1);
    }
    if (!caption2.isEmpty()) {
        QRect R(w/2, y, w/2, 50);
        QFont f("Lucida Sans Unicode", 10, QFont::Normal);
        for (float pt = 10; pt >= 5; pt -= 0.25f) {
            f.setPointSizeF(pt);
            QFontMetrics m(f, &img);
            QRect br = m.boundingRect(R, Qt::AlignCenter, caption2);
            if ((br.width() <= w/2) && (br.height() <= 45)) break;
        }
        p.setFont(f);
        p.drawText(R, Qt::AlignCenter, caption2);
        //p.setFont(QFont("Lucida Sans Unicode", 10, QFont::Normal));
        //p.drawText(QRect(w/2, y, w/2, 40), Qt::AlignCenter, caption2);
    }
    setImage(img);
}


void ImageView2::setImageV(const QString& title, const QImage& image1, const QString& caption1,
                                                const QImage& image2, const QString& caption2,
                                                const QSize& size) {
    int ph = 30;
    if (!title.isEmpty()) ph += 50;
    if (!caption1.isEmpty()) ph += 50;
    if (!caption2.isEmpty()) ph += 50;
    int w, h;
    if (size.isValid()) {
        w = size.width();
        h = size.height();
    } else {
        w = qMax(image1.width(), image2.width()) + 20;
        h = image1.height() + image2.height() + ph + 10;
    }

    int cw = w - 20;
    int ch = (h - ph) / 2;

    QImage img(w, h, QImage::Format_RGB32);
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);
    p.setPen(QPen(Qt::black));
    img.fill(0xffffffff);
    int x = 10;
    int y = 10;

    if (!title.isEmpty()) {
        p.setFont(QFont("Lucida Sans Unicode", 12, QFont::Normal));
        p.drawText(QRect(0, y, w, 40), Qt::AlignCenter, title);
        y += 50;
    }

    {
        int iw = qMin(cw, image1.width());
        int ih = qMin(ch, image1.height());
        int sx = (image1.width() - iw) / 2;
        int sy = (image1.height() - ih) / 2;
        int dx = qMax(0, (cw - iw) / 2);
        int dy = qMax(0, (ch - ih) / 2);
        p.drawImage(x+dx, y+dy, image1.copy(sx, sy, iw, ih));
        y += ch;
    }
    if (!caption1.isEmpty()) {
        QRect R(0, y, w, 50);
        QFont f("Lucida Sans Unicode", 10, QFont::Normal);
        for (float pt = 10; pt >= 5; pt -= 0.25f) {
            f.setPointSizeF(pt);
            QFontMetrics m(f, &img);
            QRect br = m.boundingRect(R, Qt::AlignCenter, caption1);
            if ((br.width() <= w/2) && (br.height() <= 45)) break;
        }
        p.setFont(f);
        p.drawText(R, Qt::AlignCenter, caption1);
        //p.setFont(QFont("Lucida Sans Unicode", 10, QFont::Normal));
        //p.drawText(QRect(0, y, w/2, 40), Qt::AlignCenter, caption1);
        y += 50;
    }

    y += 10;

    {
        int iw = qMin(cw, image2.width());
        int ih = qMin(ch, image2.height());
        int sx = (image2.width() - iw) / 2;
        int sy = (image2.height() - ih) / 2;
        int dx = qMax(0, (cw - iw) / 2);
        int dy = qMax(0, (ch - ih) / 2);
        p.drawImage(x+dx, y+dy, image2.copy(sx, sy, iw, ih));
        y += ch;
    }

    if (!caption2.isEmpty()) {
        QRect R(0, y, w, 50);
        QFont f("Lucida Sans Unicode", 10, QFont::Normal);
        for (float pt = 10; pt >= 5; pt -= 0.25f) {
            f.setPointSizeF(pt);
            QFontMetrics m(f, &img);
            QRect br = m.boundingRect(R, Qt::AlignCenter, caption2);
            if ((br.width() <= w/2) && (br.height() <= 45)) break;
        }
        p.setFont(f);
        p.drawText(R, Qt::AlignCenter, caption2);
        //p.setFont(QFont("Lucida Sans Unicode", 10, QFont::Normal));
        //p.drawText(QRect(w/2, y, w/2, 40), Qt::AlignCenter, caption2);
    }
    setImage(img);
}


void ImageView2::setPrintPreview(bool flag) {
    if (flag != m_printPreview) {
        m_printPreview = flag;
        printPreviewChanged(m_printPreview);
        update();
    }
}


void ImageView2::setPaperWidth(int value) {
    if (value != m_paperSize.width()) {
        m_paperSize.setWidth(value);
        paperWidthChanged(value);
        update();
    }
}


void ImageView2::setPaperHeight(int value) {
    if (value != m_paperSize.height()) {
        m_paperSize.setHeight(value);
        paperHeightChanged(value);
        update();
    }
}


void ImageView2::setShowGrid(bool value) {
    if (value != m_showGrid) {
        m_showGrid = value;
        showGridChanged(m_showGrid);
        update();
    }
}


void ImageView2::setPrintFrame(bool value) {
    if (value != m_printFrame) {
        m_printFrame = value;
        printFrameChanged(m_printFrame);
        update();
    }
}


void ImageView2::setColorCorrect(bool value) {
    if (value != m_colorCorrect) {
        m_colorCorrect = value;
        colorCorrectChanged(m_colorCorrect);
        update();
    }
}


void ImageView2::setSaturation(double value) {
    if (value != m_saturation) {
        m_saturation = value;
        saturationChanged(m_saturation);
        update();
    }
}


void ImageView2::setBrightness(double value) {
    if (value != m_brightness) {
        m_brightness = value;
        brightnessChanged(m_brightness);
        update();
    }
}


void ImageView2::setContrast(double value) {
    if (value != m_contrast) {
        m_contrast = value;
        contrastChanged(m_contrast);
        update();
    }
}


void ImageView2::setLayout(Layout value) {
    if (value != m_layout) {
        m_layout = value;
        layoutChanged(m_layout);
        update();
    }
}


void ImageView2::setGap(int value) {
    if (value != m_gap) {
        m_gap = value;
        gapChanged(m_gap);
        update();
    }
}


void ImageView2::setFrameWidth(double value) {
    if (value != m_frameWidth) {
        m_frameWidth = value;
        frameWidthChanged(m_frameWidth);
        update();
    }
}


void ImageView2::setLineWidth(double value) {
    if (value != m_lineWidth) {
        m_lineWidth = value;
        lineWidthChanged(m_lineWidth);
        update();
    }
}


void ImageView2::fitWidth() {
    setZoom(1.0 * m_paperSize.width() / image().width());
    setOriginX(0);
    update();
}


void ImageView2::fitHeight() {
    setZoom(1.0 * m_paperSize.height() / image().height());
    setOriginY(0);
    update();
}


void ImageView2::savePDF(const QString& text) {
    QSettings settings;
    QString inputPath = window()->windowFilePath();
    QString outputPath = settings.value("savename", inputPath).toString();

    QString filename;
    QFileInfo fi(inputPath);
    QFileInfo fo(outputPath);
    if (!fi.baseName().isEmpty()) {
        QFileInfo fn(fo.dir(), fi.baseName() + "-out.pdf");
        filename  = fn.absoluteFilePath();
    } else {
        filename  = fo.path();
    }

    filename = QFileDialog::getSaveFileName(this, "Save PDF", filename,
        "PDF Format (*.pdf);;All files (*.*)");
    if (filename.isEmpty()) {
        return;
    }

    QPrinter printer(QPrinter::HighResolution);
    printer.setOutputFormat(QPrinter::PdfFormat);
    printer.setOutputFileName(filename);
    printer.setPaperSize(m_paperSize, QPrinter::Point);
    printer.setResolution(72);
    printer.setFullPage(true);
    printer.setCreator("ImageView2");
    if (!text.isEmpty()) {
        qDebug() << text;
        printer.setDocName(text);
    }

    if (printer.printerState() != QPrinter::Error) {
        m_printing = true;
        QPainter p;
        p.begin(&printer);
        paint(p, m_paperSize);
        p.end();
        m_printing = false;
    }

    if (printer.printerState() == QPrinter::Error) {
        QMessageBox::critical(this, "Error", QString("Saving PDF '%1' failed!").arg(filename));
        return;
    }
    settings.setValue("savename", filename);

    /*if (!text.isEmpty()) {
        QFile f(filename);
        if (f.open(QFile::ReadOnly)) {
            QByteArray o;
            for (;;) {
                QByteArray L = f.readLine();
                o.append(L);
                if (L.startsWith("/Title")) {
                    o.append("/Subject (\xfe\xff");
                    const ushort *utf16 = text.utf16();
                    for (int i=0; i < text.size(); ++i) {
                        char part[2] = {char((*(utf16 + i)) >> 8), char((*(utf16 + i)) & 0xff)};
                        for(int j=0; j < 2; ++j) {
                            if (part[j] == '(' || part[j] == ')' || part[j] == '\\')
                                o.append('\\');
                            o.append(part[j]);
                        }
                    }
                    o.append(")\n");
                    break;
                }
            }
            o.append(f.readAll());
            f.close();
            if (f.open(QFile::WriteOnly | QFile::Truncate)) {
                f.write(o);
                f.close();
            }
        }
    }*/
    QDesktopServices::openUrl(QUrl::fromLocalFile(filename));
}


void ImageView2::paintEvent(QPaintEvent *e) {
    QSize sz = size();
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setWindow(0, 0, sz.width(), sz.height());

    p.translate(sz.width() / 2.0f, sz.height() / 2.0f);
    p.scale(m_resolution, m_resolution);
    if (m_printPreview) {
        p.translate(-m_paperSize.width()/2.0f, -m_paperSize.height()/2.0f);
        QRect B(0, 0, m_paperSize.width(), m_paperSize.height());
        p.fillRect(B, Qt::white);
        paint(p, m_paperSize);
    }  else {
        p.translate(-sz.width() / 2.0f, -sz.height() / 2.0f);
        paint(p, sz);
    }
}


/*
static void outlineRects1(QPainter& p, const QRectF& A, const QRectF& B, float px) {
    p.setPen(QPen(Qt::red, px));
    p.drawRect(A);
    QRectF R(B);
    R.adjust(px/2, px/2, -px/2, -px/2);
    p.drawRect(R);
    p.drawLine(A.right(), (A.top()+A.bottom())/2.0, R.left(), (R.top()+R.bottom())/2.0);
}
*/


static void outlineRects2(QPainter& p, const QRectF& A, const QRectF& B, float px) {
    p.setPen(QPen(Qt::red, px));
    QRectF R(B);
    R.adjust(px/2, px/2, -px/2, -px/2);
    QPainterPath P;
    if (A.right() < B.left()) {
        P.moveTo(A.left(), A.top());
        P.lineTo(A.right(), A.top());
        P.lineTo(R.left(), R.top());
        P.lineTo(R.right(), R.top());
        P.lineTo(R.right(), R.bottom());
        P.lineTo(R.left(), R.bottom());
        P.lineTo(A.right(), A.bottom());
        P.lineTo(A.left(), A.bottom());
        P.closeSubpath();
        P.moveTo(A.right(), A.top());
        P.lineTo(A.right(), A.bottom());
        //P.closeSubpath();
        P.moveTo(R.left(), R.top());
        P.lineTo(R.left(), R.bottom());
        //P.closeSubpath();
    } else {
        P.moveTo(A.right(), A.top());
        P.lineTo(A.left(), A.top());
        P.lineTo(R.right(), R.top());
        P.lineTo(R.left(), R.top());
        P.lineTo(R.left(), R.bottom());
        P.lineTo(R.right(), R.bottom());
        P.lineTo(A.left(), A.bottom());
        P.lineTo(A.right(), A.bottom());
        P.closeSubpath();
        P.moveTo(A.left(), A.top());
        P.lineTo(A.left(), A.bottom());
        P.moveTo(R.right(), R.top());
        P.lineTo(R.right(), R.bottom());
    }
    p.drawPath(P);
}


static void outlineRects3(QPainter& p, const QRectF& A, const QRectF& B, float px) {
    p.setPen(QPen(Qt::red, px));
    QRectF R(B);
    R.adjust(px/2, px/2, -px/2, -px/2);
    QPainterPath P;
    if (A.bottom() < B.top()) {
        P.moveTo(A.right(), A.top());
        P.lineTo(A.right(), A.bottom());
        P.lineTo(R.right(), R.top());
        P.lineTo(R.right(), R.bottom());
        P.lineTo(R.left(), R.bottom());
        P.lineTo(R.left(), R.top());
        P.lineTo(A.left(), A.bottom());
        P.lineTo(A.left(), A.top());
        P.closeSubpath();
        P.moveTo(A.left(), A.bottom());
        P.lineTo(A.right(), A.bottom());
        P.moveTo(R.left(), R.top());
        P.lineTo(R.right(), R.top());
    } else {
        P.moveTo(A.left(), A.bottom());
        P.lineTo(A.left(), A.top());
        P.lineTo(R.left(), R.bottom());
        P.lineTo(R.left(), R.top());
        P.lineTo(R.right(), R.top());
        P.lineTo(R.right(), R.bottom());
        P.lineTo(A.right(), A.top());
        P.lineTo(A.right(), A.bottom());
        P.closeSubpath();
        P.moveTo(A.left(), A.top());
        P.lineTo(A.right(), A.top());
        P.moveTo(R.left(), R.bottom());
        P.lineTo(R.right(), R.bottom());
    }
    p.drawPath(P);
}


void ImageView2::paint(QPainter &p, const QSizeF& sz) {
    float LINE_WIDTH = m_lineWidth;
    QSizeF LINE_SIZE = QSizeF(LINE_WIDTH, LINE_WIDTH);

    if (!m_printPreview || (m_layout == None)) {
        paint(p, sz, zoom(), origin(), activeView());
    } else {
        bool printFrame = m_printFrame;

        if (m_layout == Both_3X) {
            QSizeF sz3 = QSizeF((sz.width() - 2*m_gap) / 3.0f, sz.height());

            p.save();
            p.translate(LINE_WIDTH/2, LINE_WIDTH/2);
            m_printFrame = false;
            paint(p, sz3 - LINE_SIZE, zoom(1), origin(1), 1);
            m_printFrame = printFrame;
            p.translate(-LINE_WIDTH/2, -LINE_WIDTH/2);
            p.translate(sz3.width() + m_gap, 0);
            paint(p, sz3, zoom(0), origin(0), 0);
            p.translate(sz3.width() + m_gap, 0);
            p.translate(LINE_WIDTH/2, LINE_WIDTH/2);
            m_printFrame = false;
            paint(p, sz3 - LINE_SIZE, zoom(2), origin(2), 2);
            m_printFrame = printFrame;
            p.restore();

            p.save();
            p.setClipping(false);

            QTransform tr0 = viewTransform(sz3, zoom(0), origin(0));
            QTransform tr1 = viewTransform(sz3, zoom(1), origin(1));
            QTransform tr2 = viewTransform(sz3, zoom(2), origin(2));

            QRectF A = tr1.inverted().mapRect(QRectF(QPoint(0,0), sz3));
            A = tr0.mapRect(A);
            A.translate(sz3.width() + m_gap, 0);
            QRectF A0(0, 0, sz3.width(), sz3.height());

            QRectF C = tr2.inverted().mapRect(QRectF(QPoint(0,0), sz3));
            C = tr0.mapRect(C);
            C.translate(sz3.width() + m_gap, 0);
            QRectF C0(2*(sz3.width() + m_gap), 0, sz3.width(), sz3.height());

            outlineRects2(p, A, A0, LINE_WIDTH);
            outlineRects2(p, C, C0, LINE_WIDTH);

            p.restore();
        } else if ((m_layout == Left_2X) || (m_layout == Right_2X)) {
            QSizeF sz2 = QSizeF((sz.width() - m_gap) / 2.0f, sz.height());
            if (m_layout == Left_2X) {
                p.save();
                p.translate(LINE_WIDTH/2, LINE_WIDTH/2);
                m_printFrame = false;
                paint(p, sz2 - LINE_SIZE, zoom(1), origin(1), 1);
                m_printFrame = printFrame;
                p.translate(-LINE_WIDTH/2, -LINE_WIDTH/2);
                p.translate(sz2.width() + m_gap, 0);

                paint(p, sz2, zoom(0), origin(0), 0);
                p.restore();

                p.save();
                p.setClipping(false);
                QTransform tr = viewTransform(sz2, zoom(0), origin(0));
                QTransform tr2 = viewTransform(sz2, zoom(1), origin(1));
                QRectF A = tr2.inverted().mapRect(QRectF(QPoint(0,0), sz2));
                A = tr.mapRect(A);
                A.translate(sz2.width() + m_gap, 0);
                QRectF B(0, 0, sz2.width(), sz2.height());
                outlineRects2(p, A, B, LINE_WIDTH);
                p.restore();
            } else {
                p.save();
                paint(p, sz2, zoom(0), origin(0), 0);
                p.translate(sz2.width() + m_gap, 0);

                p.translate(LINE_WIDTH/2, LINE_WIDTH/2);
                m_printFrame = false;
                paint(p, sz2 - LINE_SIZE, zoom(1), origin(1), 1);
                m_printFrame = printFrame;
                p.restore();

                p.save();
                p.setClipping(false);
                QTransform tr = viewTransform(sz2, zoom(0), origin(0));
                QTransform tr2 = viewTransform(sz2, zoom(1), origin(1));
                QRectF A = tr2.inverted().mapRect(QRectF(QPoint(0,0), sz2));
                A = tr.mapRect(A);
                QRectF B(sz2.width() + m_gap, 0, sz2.width(), sz2.height());
                outlineRects2(p, A, B, LINE_WIDTH);
                p.restore();
            }
        } else {
            QSizeF sz2 = QSizeF(sz.width(), (sz.height() - m_gap) / 2.0f);
            if (m_layout == Top_2X) {
                p.save();
                p.translate(LINE_WIDTH/2, LINE_WIDTH/2);
                m_printFrame = false;
                paint(p, sz2 - LINE_SIZE, zoom(1), origin(1), 1);
                m_printFrame = printFrame;
                p.translate(-LINE_WIDTH/2, -LINE_WIDTH/2);
                p.translate(0, sz2.height() + m_gap);

                paint(p, sz2, zoom(0), origin(0), 0);
                p.restore();

                p.save();
                p.setClipping(false);
                QTransform tr = viewTransform(sz2, zoom(0), origin(0));
                QTransform tr2 = viewTransform(sz2, zoom(1), origin(1));
                QRectF A = tr2.inverted().mapRect(QRectF(QPoint(0,0), sz2));
                A = tr.mapRect(A);
                A.translate(0, sz2.height() + m_gap);
                QRectF B(0, 0, sz2.width(), sz2.height());
                outlineRects3(p, A, B, LINE_WIDTH);
                p.restore();
            } else {
                p.save();
                paint(p, sz2, zoom(0), origin(0), 0);
                p.translate(0, sz2.height() + m_gap);

                p.translate(LINE_WIDTH/2, LINE_WIDTH/2);
                m_printFrame = false;
                paint(p, sz2 - LINE_SIZE, zoom(1), origin(1), 1);
                m_printFrame = printFrame;
                p.restore();

                p.save();
                p.setClipping(false);
                QTransform tr = viewTransform(sz2, zoom(0), origin(0));
                QTransform tr2 = viewTransform(sz2, zoom(1), origin(1));
                QRectF A = tr2.inverted().mapRect(QRectF(QPoint(0,0), sz2));
                A = tr.mapRect(A);
                QRectF B(0, sz2.height() + m_gap, sz2.width(), sz2.height());
                outlineRects3(p, A, B, LINE_WIDTH);
                p.restore();
            }
        }

        //m_layoutIndex = layoutIndex;
        m_printFrame = printFrame;
    }
}


void ImageView2::paint(QPainter &p, const QSizeF& sz, double zoom, const QPointF& origin, int pass) {
    if (imageSize().isEmpty()) return;
    p.save();

    QTransform tr = viewTransform(sz, zoom, origin);
    QTransform invTr = tr.inverted();

    QRectF CR;
    if (m_printFrame)
        CR = QRectF(m_frameWidth/2, m_frameWidth/2, sz.width()-m_frameWidth, sz.height()-m_frameWidth);
    else
        CR = QRectF(0, 0, sz.width(), sz.height());
    QRectF R = invTr.mapRect(CR);

    p.save();
    if (!R.isEmpty()) {
        p.setTransform(tr, true);
        p.setClipRect(R);
        if (m_handler) {
            try {
                m_handler->draw(this, p, pass);
            }
            catch ( std::exception& e ) {
                qWarning() << e.what();
            }
        } else {
            draw(p, image(), pass);
        }
    }
    p.restore();

    if (m_printPreview && !m_printing) {
        p.save();
        p.translate(sz.width()/2.0f, sz.height()/2.0f);
        p.setPen(QPen(Qt::green));
        p.drawLine(-3, 3, 3, -3);
        p.drawLine(-3, -3, 3, 3);
        p.restore();
    }

    if (m_printFrame) {
        p.setClipping(false);
        p.setPen(QPen(Qt::black, m_frameWidth, Qt::SolidLine, Qt::SquareCap, Qt::MiterJoin));
        p.drawRect(CR);
    }

    p.restore();
}


void ImageView2::draw(QPainter& p, const QImage& image, int pass) {
    QRect aR = p.clipBoundingRect().toAlignedRect().intersected(image.rect());
    if (m_colorCorrect && (pass > 0)) {
        QImage tmp(aR.width(), aR.height(), QImage::Format_RGB32);
        for (int j = 0; j < tmp.height(); ++j) {
            for (int i = 0; i < tmp.width(); ++i) {
                QColor c = QColor::fromRgb(image.pixel(aR.x() + i, aR.y() + j)).toHsl();
                tmp.setPixel(i, j, QColor::fromHslF(
                    c.hslHueF(),
                    qBound(0.0, c.hslSaturationF() * m_saturation, 1.0),
                    qBound(0.0, c.lightnessF()* m_contrast + m_brightness, 1.0)
                ).rgb());
            }
        }
        p.drawImage(aR.x(), aR.y(), tmp);
    } else {
        p.drawImage(aR.x(), aR.y(), image.copy(aR));
    }

    if (m_showGrid && (pass > 0)) {
        QLineF L = p.deviceTransform().map(QLineF(0, 0, 1, 1));
        double qx = qMax(L.dx(), L.dy());
        if (qx > 2) {
            QLineF L = p.deviceTransform().inverted().map(QLineF(0, 0, 1, 1));
            double px = qMax(L.dx(), L.dy());
            p.setPen(QPen(Qt::gray, 0.25*px));
            for (int i = aR.left(); i <= aR.right(); ++i) {
                QPointF q0(i, aR.top());
                QPointF q1(i, aR.bottom()+1);
                p.drawLine(q0,q1);
            }
            for (int i = aR.top(); i <=  aR.bottom(); ++i) {
                QPointF q0(aR.left(), i);
                QPointF q1(aR.right()+1, i);
                p.drawLine(q0,q1);
            }
        }
    }
}
