INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR} )

FILE(GLOB_RECURSE sources *.cpp *.h *.ui *.qrc *.cu)
QT4_AUTO_WRAP(sources ${sources})
SOURCE_GROUP_BY_PATH(${sources})
CUDA_ADD_EXECUTABLE(main ${sources})

TARGET_LINK_LIBRARIES(main liboz)
