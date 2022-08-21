cmake_minimum_required(VERSION 3.4...3.18)

# ============================================================================
# ---- Directory Configuration -----------------------------------------------
# ============================================================================

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ============================================================================
# ---- Configuration ---------------------------------------------------------
# ============================================================================

project(blackthorn LANGUAGES CXX)
include(cmake/StandardProjectSettings.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  option(ENABLE_BUILD_WITH_TIME_TRACE
         "Enable -ftime-trace to generate time tracing .json files on clang"
         OFF)
  if(ENABLE_BUILD_WITH_TIME_TRACE)
    target_compile_options(project_options INTERFACE -ftime-trace)
  endif()
endif()

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# Add linker configuration
include(cmake/Linker.cmake)
configure_linker(project_options)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

# ============================================================================
# ---- Build External Libraries ----------------------------------------------
# ============================================================================

add_subdirectory(thirdparty)

# ============================================================================
# ---- Build Core Library ----------------------------------------------------
# ============================================================================

add_subdirectory(libbt)

# ============================================================================
# ---- Build Extensions Modules ----------------------------------------------
# ============================================================================

find_package(Python COMPONENTS Interpreter Development)
add_subdirectory(pybind11)
# find_package(pybind11 REQUIRED)

include_directories("${CMAKE_SOURCE_DIR}/lib/include")
set(BT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/blackthorn/src")

pybind11_add_module(
  rh_neutrino
  ${SOURCES}
  "${BT_SOURCE_DIR}/RhNeutrino.cpp"
  "${BT_SOURCE_DIR}/RhNeutrinoGeV/RhNeutrinoGeV.cpp"
  "${BT_SOURCE_DIR}/RhNeutrinoGeV/Spectra.cpp"
  "${BT_SOURCE_DIR}/RhNeutrinoGeV/Widths.cpp"
  "${BT_SOURCE_DIR}/RhNeutrinoMeV/RhNeutrinoMeV.cpp"
  "${BT_SOURCE_DIR}/RhNeutrinoMeV/Spectra.cpp"
  "${BT_SOURCE_DIR}/RhNeutrinoMeV/Widths.cpp")
target_link_libraries(rh_neutrino PRIVATE bt)
set_target_properties(
  rh_neutrino
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/blackthorn"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/blackthorn"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/blackthorn")

pybind11_add_module(fields ${SOURCES} "${BT_SOURCE_DIR}/Fields.cpp")
target_link_libraries(fields PRIVATE bt)
set_target_properties(
  fields
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/blackthorn"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/blackthorn"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/blackthorn")

pybind11_add_module(constants ${SOURCES} "${BT_SOURCE_DIR}/Constants.cpp")
target_link_libraries(constants PRIVATE bt)
set_target_properties(
  constants
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/blackthorn"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/blackthorn"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/blackthorn")
