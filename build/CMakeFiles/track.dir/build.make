# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zn/detect_tracking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zn/detect_tracking/build

# Include any dependencies generated for this target.
include CMakeFiles/track.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/track.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/track.dir/flags.make

CMakeFiles/track.dir/track.cpp.o: CMakeFiles/track.dir/flags.make
CMakeFiles/track.dir/track.cpp.o: ../track.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zn/detect_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/track.dir/track.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/track.dir/track.cpp.o -c /home/zn/detect_tracking/track.cpp

CMakeFiles/track.dir/track.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/track.dir/track.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zn/detect_tracking/track.cpp > CMakeFiles/track.dir/track.cpp.i

CMakeFiles/track.dir/track.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/track.dir/track.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zn/detect_tracking/track.cpp -o CMakeFiles/track.dir/track.cpp.s

CMakeFiles/track.dir/track.cpp.o.requires:

.PHONY : CMakeFiles/track.dir/track.cpp.o.requires

CMakeFiles/track.dir/track.cpp.o.provides: CMakeFiles/track.dir/track.cpp.o.requires
	$(MAKE) -f CMakeFiles/track.dir/build.make CMakeFiles/track.dir/track.cpp.o.provides.build
.PHONY : CMakeFiles/track.dir/track.cpp.o.provides

CMakeFiles/track.dir/track.cpp.o.provides.build: CMakeFiles/track.dir/track.cpp.o


# Object files for target track
track_OBJECTS = \
"CMakeFiles/track.dir/track.cpp.o"

# External object files for target track
track_EXTERNAL_OBJECTS =

track: CMakeFiles/track.dir/track.cpp.o
track: CMakeFiles/track.dir/build.make
track: /usr/local/opencv-2.4.9/lib/libopencv_videostab.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_ts.a
track: /usr/local/opencv-2.4.9/lib/libopencv_superres.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_stitching.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_contrib.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_nonfree.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_ocl.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_gpu.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_photo.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_objdetect.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_legacy.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_video.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_ml.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_calib3d.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_features2d.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_highgui.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_imgproc.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_flann.so.2.4.9
track: /usr/local/opencv-2.4.9/lib/libopencv_core.so.2.4.9
track: CMakeFiles/track.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zn/detect_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable track"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/track.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/track.dir/build: track

.PHONY : CMakeFiles/track.dir/build

CMakeFiles/track.dir/requires: CMakeFiles/track.dir/track.cpp.o.requires

.PHONY : CMakeFiles/track.dir/requires

CMakeFiles/track.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/track.dir/cmake_clean.cmake
.PHONY : CMakeFiles/track.dir/clean

CMakeFiles/track.dir/depend:
	cd /home/zn/detect_tracking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zn/detect_tracking /home/zn/detect_tracking /home/zn/detect_tracking/build /home/zn/detect_tracking/build /home/zn/detect_tracking/build/CMakeFiles/track.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/track.dir/depend

