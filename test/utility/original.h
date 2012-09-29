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


class Original : public Module {
    Q_OBJECT
    Q_CLASSINFO("startUp", "1")
    Q_CLASSINFO("category", "Utility");
    Q_CLASSINFO("description", "Display original image source");
    Q_CLASSINFO("rating", "***");
public:
    Q_INVOKABLE Original();
    virtual void process();
    virtual void draw(ImageView *view, QPainter &p, int pass);

protected:
    int m_inputIndex;
    bool auto_levels;
    double auto_levels_threshold;
};


class OriginalEx : public Module {
    Q_OBJECT
    Q_CLASSINFO("category", "Utility");
    Q_CLASSINFO("rating", "!");
public:
    Q_INVOKABLE OriginalEx();
    virtual void process();
    QString title() const;

protected:
    QString m_title;
    int m_number;
};
