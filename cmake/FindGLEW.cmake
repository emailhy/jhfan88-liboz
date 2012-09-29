# GLEW_FOUND
# GLEW_INCLUDE_DIR
# GLEW_LIBRARY
# GLEW_DLL

IF(WIN32)
    FIND_PROGRAM( GLEW_DLL glew32.dll )
    
    #IF(GLEW_DLL)
        GET_FILENAME_COMPONENT( GLEW_DIR "${GLEW_DLL}" PATH )
        GET_FILENAME_COMPONENT( GLEW_DIR "${GLEW_DIR}" PATH )
    #ENDIF()

    FIND_PATH( GLEW_INCLUDE_DIR GL/glew.h HINTS ${GLEW_DIR}/include )
    FIND_LIBRARY( GLEW_LIBRARY glew32 HINTS ${GLEW_DIR}/lib )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLEW DEFAULT_MSG 
        GLEW_INCLUDE_DIR 
        GLEW_LIBRARY
        GLEW_DLL )
ELSE()
    FIND_PATH( GLEW_INCLUDE_DIR GL/glew.h
        /usr/include
        /usr/local/include
        /sw/include
        /opt/local/include )
        
    FIND_LIBRARY( GLEW_LIBRARY
        NAMES GLEW glew
        PATHS
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/local/lib )
        
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLEW DEFAULT_MSG 
        GLEW_INCLUDE_DIR 
        GLEW_LIBRARY )
ENDIF()

