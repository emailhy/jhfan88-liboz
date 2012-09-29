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
#pragma once

class GLView : public QGLWidget {
    Q_OBJECT
    Q_PROPERTY(double zoom READ zoom WRITE setZoom NOTIFY zoomChanged(double)) 
    Q_PROPERTY(double originX READ originX WRITE setOriginX NOTIFY originXChanged)
    Q_PROPERTY(double originY READ originY WRITE setOriginY NOTIFY originYChanged)
public:
    GLView(QWidget *parent);
    ~GLView();

    virtual void restoreSettings(QSettings& settings);
    virtual void saveSettings(QSettings& settings);

    QPointF origin() const { return m_origin; }
    double originX() const { return m_origin.x(); }
    double originY() const { return m_origin.y(); }
    double zoom() const { return m_zoom; }

    QPointF view2canvas(const QPointF& p) const;
    QPointF canvas2view(const QPointF& p) const;
    QTransform viewTransform() const;
    float pt2px(float pt) const;

    struct Handler {
        virtual QSizeF canvasSize();
        virtual void paintGL(GLView *view, const QRectF& R);
        virtual void dragBegin(GLView *view, QMouseEvent *e);
        virtual void dragMove(GLView *view, QMouseEvent *e, const QPoint& start);
        virtual void dragEnd(GLView *view, QMouseEvent *e, const QPoint& start);
    };
    void setHandler(Handler *handler);

public slots:
    void setOriginX(double value);
    void setOriginY(double value);
    void setZoom(double value);
    void zoomIn();
    void zoomOut();
    void reset();

signals:
    void originXChanged(double);
    void originYChanged(double);
    void zoomChanged(double);

protected:
    virtual void initializeGL();
    virtual QSizeF canvasSize() const;
    virtual void paintGL();
    virtual void paintGL(const QRectF& R);

    virtual bool eventFilter( QObject *watched, QEvent *e );
    virtual void leaveEvent( QEvent *e );
    virtual void keyPressEvent( QKeyEvent *e );
    virtual void keyReleaseEvent( QKeyEvent *e );
    virtual void mousePressEvent( QMouseEvent *e );
    virtual void mouseMoveEvent( QMouseEvent *e );
    virtual void mouseReleaseEvent( QMouseEvent *e );
    virtual void wheelEvent(QWheelEvent *e);
    virtual void mouseDragBegin(QMouseEvent *e);
    virtual void mouseDragMove(QMouseEvent *e, const QPoint& start);
    virtual void mouseDragEnd(QMouseEvent *e, const QPoint& start);

protected:
    QPointF m_origin;
    double m_zoom;
    bool m_spacePressed;
    
    enum Mode { MODE_NONE=0, MODE_PAN, MODE_DRAG };
    Mode m_mode;
    QPoint m_dragStart;
    Qt::MouseButton m_dragButton;
    QPointF m_dragOrigin;
    QCursor m_cursor;
    Handler *m_handler;
};
