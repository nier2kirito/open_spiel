# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nierxkrito/open_spiel/open_spiel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nierxkrito/open_spiel/open_spiel/python

# Include any dependencies generated for this target.
include abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/compiler_depend.make

# Include the progress variables for this target.
include abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/progress.make

# Include the compile flags for this target's objects.
include abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/flags.make

abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o: abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/flags.make
abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o: /home/nierxkrito/open_spiel/open_spiel/abseil-cpp/absl/strings/internal/cordz_info.cc
abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o: abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/nierxkrito/open_spiel/open_spiel/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o"
	cd /home/nierxkrito/open_spiel/open_spiel/python/abseil-cpp/absl/strings && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o -MF CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o.d -o CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o -c /home/nierxkrito/open_spiel/open_spiel/abseil-cpp/absl/strings/internal/cordz_info.cc

abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.i"
	cd /home/nierxkrito/open_spiel/open_spiel/python/abseil-cpp/absl/strings && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nierxkrito/open_spiel/open_spiel/abseil-cpp/absl/strings/internal/cordz_info.cc > CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.i

abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.s"
	cd /home/nierxkrito/open_spiel/open_spiel/python/abseil-cpp/absl/strings && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nierxkrito/open_spiel/open_spiel/abseil-cpp/absl/strings/internal/cordz_info.cc -o CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.s

# Object files for target absl_cordz_info
absl_cordz_info_OBJECTS = \
"CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o"

# External object files for target absl_cordz_info
absl_cordz_info_EXTERNAL_OBJECTS =

abseil-cpp/absl/strings/libabsl_cordz_info.a: abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o
abseil-cpp/absl/strings/libabsl_cordz_info.a: abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/build.make
abseil-cpp/absl/strings/libabsl_cordz_info.a: abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/nierxkrito/open_spiel/open_spiel/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libabsl_cordz_info.a"
	cd /home/nierxkrito/open_spiel/open_spiel/python/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -P CMakeFiles/absl_cordz_info.dir/cmake_clean_target.cmake
	cd /home/nierxkrito/open_spiel/open_spiel/python/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/absl_cordz_info.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/build: abseil-cpp/absl/strings/libabsl_cordz_info.a
.PHONY : abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/build

abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/clean:
	cd /home/nierxkrito/open_spiel/open_spiel/python/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -P CMakeFiles/absl_cordz_info.dir/cmake_clean.cmake
.PHONY : abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/clean

abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/depend:
	cd /home/nierxkrito/open_spiel/open_spiel/python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nierxkrito/open_spiel/open_spiel /home/nierxkrito/open_spiel/open_spiel/abseil-cpp/absl/strings /home/nierxkrito/open_spiel/open_spiel/python /home/nierxkrito/open_spiel/open_spiel/python/abseil-cpp/absl/strings /home/nierxkrito/open_spiel/open_spiel/python/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/depend

