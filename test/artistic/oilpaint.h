// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class OilPaint : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Artistic");
    Q_CLASSINFO("description", "Simple oil paint. Based on Daniel Müller's flowpaint implementation.");
public:
    Q_INVOKABLE OilPaint();
    virtual void process();

    double m_st_sigma;
    double m_flow_blur;
    double m_bump_scale;
    double m_phong_specular;
    double m_phong_shininess;
};
