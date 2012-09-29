#  MATLAB_FOUND - system has matlab libraries
#  MATLAB_INCLUDE_DIR - the matlab include directory
#  MATLAB_LIBRARIES - link these to use matlab

IF(WIN32)
    SET(MATLAB_ROOT "C:/Program Files/MATLAB/R2011a")
ENDIF(WIN32)

FIND_PATH(MATLAB_INCLUDE_DIR "mex.h" HINTS "${MATLAB_ROOT}/extern/include")
FIND_LIBRARY(MATLAB_mx_LIBRARY libmx HINTS "${MATLAB_ROOT}//extern/lib/win64/microsoft")
FIND_LIBRARY(MATLAB_mat_LIBRARY libmat HINTS "${MATLAB_ROOT}//extern/lib/win64/microsoft")

FIND_PACKAGE_HANDLE_STANDARD_ARGS(MATLAB DEFAULT_MSG
    MATLAB_INCLUDE_DIR
    MATLAB_mx_LIBRARY
    MATLAB_mat_LIBRARY )

IF(MATLAB_FOUND)
    SET(MATLAB_LIBRARIES
        ${MATLAB_mx_LIBRARY}
        ${MATLAB_mat_LIBRARY})
ENDIF()
