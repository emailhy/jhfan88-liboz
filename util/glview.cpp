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
#include "glview.h"


GLView::GLView(QWidget *parent) : QGLWidget(parent) {
    m_spacePressed = false;
    m_mode = MODE_NONE;
    m_handler = 0;
    m_zoom = 1;
    qApp->installEventFilter(this);
}


GLView::~GLView() {
}


void GLView::restoreSettings(QSettings& settings) {
    setZoom(settings.value("zoom", 1.0).toDouble());
    setOriginX(settings.value("originX", 0).toFloat());
    setOriginY(settings.value("originY", 0).toFloat());
}


void GLView::saveSettings(QSettings& settings) {
    settings.setValue("zoom", m_zoom);
    settings.setValue("originX", m_origin.x());
    settings.setValue("originY", m_origin.y());
}


QPointF GLView::view2canvas(const QPointF& p) const {
    QSize sz = size();
    float z = m_zoom;
    QSizeF cz = canvasSize();
    return QPointF(
        (p.x() - m_origin.x() * z - sz.width() / 2) / z + cz.width() / 2,
        (p.y() - m_origin.y() * z - sz.height() / 2) / z + cz.height() / 2
    );
}


QPointF GLView::canvas2view(const QPointF& q) const {
    QSize sz = size();
    float z = m_zoom;
    QSizeF cz = canvasSize();
    return QPointF(
        (q.x() - cz.width() / 2) * z + m_origin.x() * z + sz.width() / 2,
        (q.y() - cz.height() / 2) * z + m_origin.y() * z + sz.height() / 2
    );
}


QTransform GLView::viewTransform() const {
    QSize sz = size();
    double z = zoom();
    QPointF o = origin();
    QSizeF cz =  canvasSize();
    QTransform tr;
    tr.translate(sz.width()/2, sz.height()/2);
    tr.translate(o.x() * z, o.y() * z);
    tr.scale(z, z);
    tr.translate(-cz.width()/2, -cz.height()/2);
    return tr;
}


float GLView::pt2px(float pt) const {
    return pt / m_zoom;
}


void GLView::setHandler(Handler *handler) {
    m_handler = handler;
    update();
}


// void GLView::setOverlay(QWidget *overlay) {
//     if (overlay) {
//         setMouseTracking(true);
//         m_overlay = overlay;
//         m_overlay->setVisible(false);
//     } else {
//         setMouseTracking(false);
//         if (m_overlay) m_overlay->setVisible(false);
//         m_overlay = 0;
//     }
// }


void GLView::setOriginX(double value) {
    double x = value;
    if (m_origin.x() != x) {
        m_origin.setX(x);
        originXChanged(x);
        updateGL();
    }
}


void GLView::setOriginY(double value) {
    double y = value;
    if (m_origin.y() != y) {
        m_origin.setY(y);
        originYChanged(y);
        updateGL();
    }
}


void GLView::setZoom(double value) {
    if (value != m_zoom) {
        m_zoom = value;
        zoomChanged(m_zoom);
        updateGL();
    }
}


void GLView::zoomIn() {
    setZoom(m_zoom * 2);
}


void GLView::zoomOut() {
    setZoom(m_zoom / 2);
}


void GLView::reset() {
    m_zoom = 1.0;
    m_origin = QPoint(0, 0);
    zoomChanged(1);
    originXChanged(0);
    originYChanged(0);
    update();
}


void GLView::initializeGL() {
    //glewInit();
}

    
QSizeF GLView::canvasSize() const {
    if (m_handler) {
        return m_handler->canvasSize();
    }
    return QSizeF();
}


void GLView::paintGL() {
    QSize sz = size();
    glViewport(0, 0, sz.width(), sz.height());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, sz.width(), sz.height(), 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    QTransform  tr = viewTransform();
    QTransform invTr = tr.inverted();
    QRectF R = invTr.mapRect(QRectF(QPointF(0,0), sz));

    float m[4][4];
    memset(m, 0, 16*sizeof(float));
    m[0][0] = tr.m11();
    m[0][1] = tr.m12();
    m[1][0] = tr.m21();
    m[1][1] = tr.m22();
    m[3][0] = tr.dx();
    m[3][1] = tr.dy();
    m[2][2] = m[3][3] = 1;
    glMultMatrixf((float*)m);

    paintGL(R);
}


void GLView::paintGL(const QRectF& R) {
    if (m_handler) {
        m_handler->paintGL(this, R);
    }
}


bool GLView::eventFilter( QObject *watched, QEvent *e ) {
    if (!hasFocus() && (m_mode == MODE_NONE) && (e->type() == QEvent::KeyPress)) {
        QKeyEvent *k = (QKeyEvent*)e;
        if (k->key() == Qt::Key_Space) {
            QPoint gp = mapFromGlobal(QCursor::pos());
            if (rect().contains(gp)) {
                setFocus(Qt::OtherFocusReason);
                keyPressEvent(k);
                return true;
            }
        }
    }
    return QWidget::eventFilter(watched, e);
}


void GLView::leaveEvent( QEvent *e ) {
    if (m_spacePressed) {
        setCursor(m_cursor);
        m_cursor = QCursor();
        m_spacePressed = false;
    }
    //if (m_overlay) {
    //    m_overlay->setVisible(false);
    //}
    QWidget::leaveEvent(e);
}


void GLView::keyPressEvent( QKeyEvent *e ) {
    if ((m_mode == MODE_NONE) && (e->key() == Qt::Key_Space)) {
        if (!m_spacePressed) {
            //if (m_overlay && m_overlay->isVisible()) m_overlay->setVisible(false);
            m_cursor = cursor();
            setCursor(Qt::OpenHandCursor);
            m_spacePressed = true;
        }
    } else {
        QWidget::keyPressEvent(e);
    }
}


void GLView::keyReleaseEvent( QKeyEvent *e ) {
    if (e->key() == Qt::Key_Space) {
        if (!e->isAutoRepeat()) {
            if (m_mode == MODE_NONE) {
                setCursor(m_cursor);
                m_cursor = QCursor();
            }
            m_spacePressed = false;
        }
    } else {
        QWidget::keyReleaseEvent(e);
    }
}


void GLView::mousePressEvent( QMouseEvent *e ) {
    if ((m_mode == MODE_NONE) && (e->button() != Qt::NoButton)) {
        if (!m_spacePressed || (e->buttons() != Qt::LeftButton)) {
            m_mode = MODE_DRAG;
            m_dragStart =  e->pos();
            m_dragButton = e->button();
            mouseDragBegin(e);
            if (e->isAccepted()) return;
        } 
        setCursor(Qt::ClosedHandCursor);
        m_mode = MODE_PAN;
        m_dragStart = e->pos();
        m_dragButton = e->button();
        m_dragOrigin = m_origin;
        update();
    }
}


void GLView::mouseMoveEvent( QMouseEvent *e ) {
    if (m_mode == MODE_PAN) {
        m_origin = m_dragOrigin + QPointF(e->pos() - m_dragStart) / m_zoom;
        originXChanged(m_origin.x());
        originYChanged(m_origin.y());
        update();
    } else if (m_mode == MODE_DRAG) {
        mouseDragMove(e, m_dragStart);
    }
    //if (m_overlay && !m_spacePressed && (m_mode != MODE_PAN)) {
    //    QSize sz = m_overlay->size();
    //    m_overlay->move(e->pos().x() - sz.width()/2, e->pos().y() - sz.height()/2);
    //    if (m_overlay->isHidden()) m_overlay->setVisible(true);
    //}
}


void GLView::mouseReleaseEvent( QMouseEvent *e ) {
    if ((m_mode == MODE_PAN) && (e->button() == m_dragButton)) {
        if (m_spacePressed) {
            setCursor(QCursor(Qt::OpenHandCursor));
        } else {
            setCursor(m_cursor);
            m_cursor = QCursor();
        }
        m_mode = MODE_NONE;
        update();
    } else if ((m_mode == MODE_DRAG) && (e->button() == m_dragButton)) {
        m_mode = MODE_NONE;
        mouseDragEnd(e, m_dragStart);
    }
}


void GLView::wheelEvent(QWheelEvent *e) {
    QSize sz = size();
    double u = e->delta() / 120.0 / 4.0;
    if (u < -0.5) u = -0.5;
    if (u > 0.5) u = 0.5;
    setZoom(m_zoom * (1 + u));
}


void GLView::mouseDragBegin(QMouseEvent *e) {
    if (m_handler) 
        m_handler->dragBegin(this, e);
    else
        e->ignore();
}


void GLView::mouseDragMove(QMouseEvent *e, const QPoint& start) {
    if (m_handler) 
        m_handler->dragMove(this, e, start);
    else
        e->ignore();
}


void GLView::mouseDragEnd(QMouseEvent *e, const QPoint& start) {
    if (m_handler) 
        m_handler->dragEnd(this, e, start);
    else
        e->ignore();
}


QSizeF GLView::Handler::canvasSize() {
    return QSizeF();
}


void GLView::Handler::paintGL(GLView *view, const QRectF& R) {
}


void GLView::Handler::dragBegin(GLView *view, QMouseEvent *e) {
    e->ignore();
}


void GLView::Handler::dragMove(GLView *view, QMouseEvent *e, const QPoint& start) {
    e->ignore();
}


void GLView::Handler::dragEnd(GLView *view, QMouseEvent *e, const QPoint& start) {
    e->ignore();
}
