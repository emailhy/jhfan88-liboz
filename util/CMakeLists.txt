CONFIGURE_FILE( settingscheck_version.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/settingscheck_version.h @ONLY )

SET( QT_USE_QTOPENGL 1 )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${CAIRO_INCLUDE_DIR} )

IF(LIBAV_FOUND)
    ADD_DEFINITIONS(-DHAVE_LIBAV)
ENDIF()

FILE( GLOB_RECURSE sources *.cpp *.h *.ui *.qrc )
QT4_AUTO_WRAP( sources ${sources} )
CUDA_ADD_LIBRARY( util STATIC ${sources} )

SOURCE_GROUP( src REGULAR_EXPRESSION "c$|cpp$|hpp$|h$|ui$|qrc$|cu$" )
SOURCE_GROUP( generated REGULAR_EXPRESSION "cxx$|ui_" )

IF( MSVC )
    SET_TARGET_PROPERTIES( util
        PROPERTIES
        COMPILE_FLAGS "/FIstable.h /Yustable.h")

    SET_SOURCE_FILES_PROPERTIES(stable.cpp
        PROPERTIES
        COMPILE_FLAGS "/Ycstable.h")

ELSEIF( XCODE_VERSION )
#    SET_TARGET_PROPERTIES( util
#        PROPERTIES
#        XCODE_ATTRIBUTE_GCC_PRECOMPILE_PREFIX_HEADER YES )
    SET_TARGET_PROPERTIES( util
        PROPERTIES
#        XCODE_ATTRIBUTE_GCC_PREFIX_HEADER stable.h
        COMPILE_FLAGS "-include stable.h -Winvalid-pch" )

ELSEIF( CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stable.h
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/stable.h ${CMAKE_CURRENT_BINARY_DIR}/stable.h
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/stable.h
    )
    ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stable.h.gch
        COMMAND ${CMAKE_CXX_COMPILER} $(CXX_DEFINES) $(CXX_FLAGS) -x c++-header
                -o ${CMAKE_CURRENT_BINARY_DIR}/stable.h.gch
                ${CMAKE_CURRENT_BINARY_DIR}/stable.h
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stable.h
    )
    SET_SOURCE_FILES_PROPERTIES(${sources} PROPERTIES OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stable.h.gch)
    SET_TARGET_PROPERTIES( util
        PROPERTIES
        COMPILE_FLAGS "-include stable.h -Winvalid-pch" )

ELSE()
    MESSAGE(FATAL_ERROR "Unsupported compiler! ${CMAKE_CXX_COMPILER_ID}")
ENDIF()
