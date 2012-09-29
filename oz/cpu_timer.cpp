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
#include <oz/cpu_timer.h>
#if defined WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined __MACH__ && defined __APPLE__
#include <mach/mach_time.h>
#elif defined __linux || defined __linux__
#include <time.h>
#else
#include <sys/time.h>
#endif


oz::cpu_timer::cpu_timer() {
    reset();
}


void oz::cpu_timer::reset() {
    m_current = now();
}


double oz::cpu_timer::now() {
#if defined WIN32

    static LONGLONG s_freq = 0;
    if (s_freq == 0) {
        LARGE_INTEGER li;
        QueryPerformanceFrequency(&li);
        s_freq = li.QuadPart;
    }
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (1000.0 * li.QuadPart) / s_freq;

#elif defined __linux || defined __linux__

    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return 1000.0 * tp.tv_sec + tp.tv_nsec / 1e6;

#elif defined __MACH__ && defined __APPLE__

    static double s_freq = 0;
    if( s_freq == 0 ) {
        mach_timebase_info_data_t ti;
        mach_timebase_info(&ti);
        s_freq = ti.denom * 1e9 / ti.numer;
    }
    return (double)mach_absolute_time() / s_freq;

#else

    timeval x;
    gettimeofday(&x, 0);
    return 1000.0 * x.tv_sec + x.tv_usec / 1000.0;

#endif
}


double oz::cpu_timer::elapsed_time( bool reset ) {
    double x = now();
    double dt = x - m_current;
    if (reset) {
        m_current = x;
    }
    return dt;
}
