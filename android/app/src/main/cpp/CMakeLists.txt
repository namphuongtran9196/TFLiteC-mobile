cmake_minimum_required(VERSION 3.10.2)
set(CMAKE_CXX_STANDARD 14)
set(TARGET_NAME tfliteC)

file(GLOB srcs ${CMAKE_SOURCE_DIR}/*.c ${CMAKE_SOURCE_DIR}/*.cpp )
file(GLOB hdrs ${CMAKE_SOURCE_DIR}/*.h ${CMAKE_SOURCE_DIR}/*.hpp )

#OpenCV
######################################################
# Set OpenCV library here
set(OPENCV_ANDROID "${CMAKE_CURRENT_LIST_DIR}/opencv-4.3.0-android-sdk/OpenCV-android-sdk")
set(OpenCV_STATIC on)
set(OpenCV_DIR ${OPENCV_ANDROID}/sdk/native/jni)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
######################################################

# TFLite
######################################################
add_library(tflite SHARED IMPORTED)
set_target_properties( tflite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/${ANDROID_ABI}/libtensorflowlite.so)

# Specify where to find the header files for TF Lite C++
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/tensorflow)

# Specify where to find the header files for Flatbuffers C++
include_directories(${CMAKE_CURRENT_LIST_DIR}/flatbuffers/include)
######################################################

add_library(
        ${TARGET_NAME} SHARED
        ${hdrs}
        ${srcs}
)

find_library(
        log-lib
        log)

find_library(jnigraphics-lib jnigraphics)

target_link_libraries(
        ${TARGET_NAME}
        android
        jnigraphics
        ${OpenCV_LIBS}
        ${log-lib}
        tflite)

