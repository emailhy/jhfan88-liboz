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

#include <oz/config.h>
#include <limits>

namespace oz {

    template<typename T> class err_stat_t {
    public:
        err_stat_t() {
            N_ = 0;
            min_ = std::numeric_limits<T>::max();
            max_ = -min_;
            mean_ = mean2_ = 0;
        }

        void operator+=( T x ) {
            if ( x < min_) min_ = x;
            if ( x > max_) max_ = x;
            N_++;
            T delta = x - mean_;
            mean_ += delta / N_;
            mean2_ += delta * (x - mean_);
        }

        T min() const { return min_; }
        T max() const { return max_; }
        T mean() const { return mean_; }
        T sigma() const { return sqrt(mean2_ / N_); }

    private:
        size_t N_;
        T min_;
        T max_;
        T mean_;
        T mean2_;

    };

}
