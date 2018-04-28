cmake_minimum_required( VERSION 2.8 )
project( tracking )

set( CMAKE_CXX_FLAGS "-std=c++11" )
set( CMAKE_BUILD_TYPE "Release" )
set(OpenCV_DIR "/usr/local/opencv-2.4.9/share/OpenCV")

find_package( OpenCV 2.4.9 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( detect detect.cpp )
target_link_libraries( detect ${OpenCV_LIBS} )

add_executable( track track.cpp )
target_link_libraries( track ${OpenCV_LIBS} )

add_executable( detect_track detect_track.cpp )
target_link_libraries( detect_track ${OpenCV_LIBS} )
