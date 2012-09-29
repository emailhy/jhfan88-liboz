// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class ConvPyrTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
public:
    Q_INVOKABLE ConvPyrTest();
    virtual void process();

protected:
};

