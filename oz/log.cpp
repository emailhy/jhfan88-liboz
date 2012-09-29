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
#include <oz/log.h>
#include <oz/cpu_image.h>
#include <oz/gpu_image.h>
#include "qdebug.h"
#ifdef WIN32
#include <windows.h>
#include <DbgHelp.h>
#endif


void oz::log_debug( const char *format, ... ) {
   va_list args;
   char buffer[1024];
   va_start( args, format );
#ifdef _MSC_VER
   vsprintf_s( buffer, 1024, format, args );
#else
   vsnprintf( buffer, 1024, format, args );
#endif
   qDebug( "%s", buffer );
}


void oz::log_warn( const char *format, ... ) {
   va_list args;
   char buffer[1024];
   va_start( args, format );
#ifdef _MSC_VER
   vsprintf_s( buffer, 1024, format, args );
#else
   vsnprintf( buffer, 1024, format, args );
#endif
    qWarning( "%s", buffer );
}


void oz::log_stack() {
#ifdef WIN32
    unsigned i;
    void* stack[10];
    unsigned short frames;
    SYMBOL_INFO* symbol;
    HANDLE process;

    process = GetCurrentProcess();
    SymInitialize( process, NULL, TRUE );
    SymSetOptions(SYMOPT_LOAD_LINES);

    frames = CaptureStackBackTrace(2, 5, stack, NULL);
    symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256, 1);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    DWORD dwDisplacement;
    IMAGEHLP_LINE64 line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    for (i = 0; i < frames; ++i) {
        SymFromAddr( process, (DWORD64)(stack[i]), 0, symbol);
        if (SymGetLineFromAddr64(process, symbol->Address, &dwDisplacement, &line)) {
            log_warn( "%s(%d): [%i] 0x%0X - %s", line.FileName, line.LineNumber, frames - i - 1, symbol->Address, symbol->Name);
        } else {
            log_warn( "%i: 0x%0X - %s", frames - i - 1, symbol->Address, symbol->Name );
        }
    }

    free( symbol );
#endif
}


static void (*g_image_callback)(const oz::cpu_image&, const char*, void*) = NULL;
static void *g_image_user = NULL;


void oz::log_image( const oz::cpu_image& image, const char* format, ... ) {
   va_list args;
   char buffer[1024];
   va_start( args, format );
#ifdef _MSC_VER
   vsprintf_s( buffer, 1024, format, args );
#else
   vsnprintf( buffer, 1024, format, args );
#endif
   if (g_image_callback) {
       g_image_callback(image, buffer, g_image_user);
   } else {
       log_warn("Image log handler not installed!");
   }
}


void oz::log_image( const oz::gpu_image& image, const char* format, ... ) {
   va_list args;
   char buffer[1024];
   va_start( args, format );
#ifdef _MSC_VER
   vsprintf_s( buffer, 1024, format, args );
#else
   vsnprintf( buffer, 1024, format, args );
#endif
    log_image(image.cpu(), "%s", buffer);
}


void oz::install_log_handler(void (*callback)(const cpu_image&, const char*, void*), void *user) {
    g_image_callback = callback;
    g_image_user = user;
}
