# CAIRO_FOUND
# CAIRO_INCLUDE_DIR
# CAIRO_LIBRARY
# CAIRO_DLL

IF(WIN32)
    FIND_PROGRAM( CAIRO_DLL libcairo2.dll )

    GET_FILENAME_COMPONENT( CAIRO_DIR "${CAIRO_DLL}" PATH )
    GET_FILENAME_COMPONENT( CAIRO_DIR "${CAIRO_DIR}" PATH )

    FIND_PATH( CAIRO_INCLUDE_DIR cairo.h HINTS ${CAIRO_DIR}/include/cairo )
    FIND_LIBRARY( CAIRO_LIBRARY libcairo2 HINTS ${CAIRO_DIR}/lib )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(CAIRO DEFAULT_MSG
        CAIRO_INCLUDE_DIR
        CAIRO_LIBRARY
        CAIRO_DLL )
ELSE()
    FIND_PATH( CAIRO_INCLUDE_DIR cairo.h
        /usr/include/cairo
        /usr/local/include/cairo )

    FIND_LIBRARY( CAIRO_LIBRARY
        NAMES cairo
        PATHS
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(CAIRO DEFAULT_MSG
        CAIRO_INCLUDE_DIR
        CAIRO_LIBRARY )
ENDIF()
