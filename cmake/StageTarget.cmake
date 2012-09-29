ADD_CUSTOM_TARGET(STAGE ALL)
IF(MSVC_IDE)
    SET(STAGE_CFG "$(ConfigurationName)")
ELSE()
    SET(STAGE_CFG "${CMAKE_BUILD_TYPE}")
ENDIF()
SET(STAGE_DIR "bin/${STAGE_CFG}")

INSTALL(FILES
    "${QT_BINARY_DIR}/QtCored4.dll" "${QT_BINARY_DIR}/QtGuid4.dll"
    DESTINATION .
    CONFIGURATIONS Debug)

INSTALL(FILES
    "${QT_BINARY_DIR}/QtCore4.dll" "${QT_BINARY_DIR}/QtGui4.dll"
    DESTINATION .
    CONFIGURATIONS Release)

ADD_CUSTOM_COMMAND(TARGET STAGE POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory ${STAGE_DIR}
    COMMAND if "${STAGE_CFG}"==Debug (
            "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtCored4.dll ${STAGE_DIR} &&
            "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtCored4.pdb ${STAGE_DIR} &&
            "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtGuid4.dll ${STAGE_DIR} &&
            "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtGuid4.pdb ${STAGE_DIR}
            ) else (
            "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtCore4.dll ${STAGE_DIR} &&
            "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtGui4.dll ${STAGE_DIR}
            )
   WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

SET(CUDA_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
IF(CMAKE_CL_64)
    SET(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
    FILE(GLOB CUDA_DLL_POSTFIX "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64*.dll")
ELSE()
    SET(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32")
    FILE(GLOB CUDA_DLL_POSTFIX "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart32*.dll")
ENDIF()
STRING(REGEX MATCH "(32|64)_[0-9]+_[0-9]+.dll$" CUDA_DLL_POSTFIX ${CUDA_DLL_POSTFIX})

FIND_PROGRAM(CUDA_CUDART_DLL cudart${CUDA_DLL_POSTFIX} HINTS "${CUDA_BIN_DIR}")
INSTALL(FILES "${CUDA_CUDART_DLL}"  DESTINATION . CONFIGURATIONS Debug Release)
FOREACH(lib npp curand cufft)
    FIND_LIBRARY(CUDA_${lib}_LIBRARY ${lib}.lib HINTS "${CUDA_LIB_DIR}")
    FIND_PROGRAM(CUDA_${lib}_DLL ${lib}${CUDA_DLL_POSTFIX} HINTS "${CUDA_BIN_DIR}")
    MARK_AS_ADVANCED(CUDA_${lib}_LIBRARY)

    IF(CUDA_${lib}_DLL)
        INSTALL(FILES ${CUDA_${lib}_DLL} DESTINATION . CONFIGURATIONS Debug Release)

        ADD_CUSTOM_COMMAND(TARGET STAGE POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${CUDA_${lib}_DLL} ${STAGE_DIR}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    ENDIF()
ENDFOREACH()

IF(OpenCV_FOUND)
    SET(OpenCV_TAG ${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH})
    FOREACH(lib core imgproc video)
        FIND_PROGRAM(OpenCV_${lib}_DEBUG_DLL opencv_${lib}${OpenCV_TAG}d.dll
            HINTS
            ${OpenCV_DIR}/bin ${OpenCV_DIR}/bin/Debug)
        FIND_PROGRAM(OpenCV_${lib}_DLL opencv_${lib}${OpenCV_TAG}.dll
            HINTS
            ${OpenCV_DIR}/bin ${OpenCV_DIR}/bin/Release)

        IF (OpenCV_${lib}_DLL AND OpenCV_${lib}_DEBUG_DLL)
            INSTALL(FILES ${OpenCV_${lib}_DEBUG_DLL} DESTINATION . CONFIGURATIONS Debug)
            INSTALL(FILES ${OpenCV_${lib}_DLL} DESTINATION . CONFIGURATIONS Release)

            ADD_CUSTOM_COMMAND(TARGET STAGE POST_BUILD
                COMMAND if $(ConfigurationName)==Debug (
                        "${CMAKE_COMMAND}" -E copy_if_different ${OpenCV_${lib}_DEBUG_DLL} ${STAGE_DIR}
                        ) else (
                        "${CMAKE_COMMAND}" -E copy_if_different ${OpenCV_${lib}_DLL} ${STAGE_DIR}
                        )
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        ENDIF()
    ENDFOREACH()
ENDIF()

IF(FFMPEG_FOUND)
    INSTALL(FILES
        ${FFmpeg_avutil_DLL} ${FFmpeg_avcodec_DLL} ${FFmpeg_avformat_DLL} ${FFmpeg_swscale_DLL}
        DESTINATION .
        CONFIGURATIONS Debug Release)

    ADD_CUSTOM_COMMAND(TARGET STAGE POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${FFmpeg_avutil_DLL}" ${STAGE_DIR} &&
                "${CMAKE_COMMAND}" -E copy_if_different "${FFmpeg_avcodec_DLL}" ${STAGE_DIR} &&
                "${CMAKE_COMMAND}" -E copy_if_different "${FFmpeg_avformat_DLL}" ${STAGE_DIR} &&
                "${CMAKE_COMMAND}" -E copy_if_different "${FFmpeg_swscale_DLL}" ${STAGE_DIR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
ENDIF()
