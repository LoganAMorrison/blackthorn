cmake_minimum_required(VERSION 3.4...3.18)

project(libbt LANGUAGES CXX Fortran)

include(FetchContent)
include(ExternalProject)

# ============================================================================
# ---- Requirements ----------------------------------------------------------
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_library(
  PYTHIA8_LIB
  NAMES pythia8
  HINTS "/usr/lib"
)
find_library(
  CUBA_LIB
  NAMES cuba
  HINTS "/usr/local/lib"
)
find_package(Boost REQUIRED)
find_package(fmt)
find_package(Threads REQUIRED)
# find_package(TBB REQUIRED)
find_package(GSL REQUIRED)
# find_package(GSL REQUIRED)

find_package(pybind11)
find_package(Python COMPONENTS Interpreter Development)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC")


# ============================================================================
# ---- Build Core Library ----------------------------------------------------
# ============================================================================

set(LIBBT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIBBT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(LIBBT_COMMON_INCLUDES ${LIBBT_INCLUDE_DIR} pybind11::pybind11 ${Python_INCLUDE_DIRS})

set(LIBBT_TOOLS_DIR "${LIBBT_SOURCE_DIR}/Tools")
set(LIBBT_TOOLS_SOURCES
  "${LIBBT_TOOLS_DIR}/Statistics.cpp" 
  "${LIBBT_TOOLS_DIR}/Printing.cpp"
  "${LIBBT_TOOLS_DIR}/Check.cpp" 
  "${LIBBT_TOOLS_DIR}/Special.cpp"
  "${LIBBT_TOOLS_DIR}/Array.cpp"
  "${LIBBT_TOOLS_DIR}/Numpy.cpp"
)

add_library(bt_tools OBJECT ${LIBBT_TOOLS_SOURCES})
target_link_libraries(bt_tools PUBLIC project_options PRIVATE ${Python_LIBRARIES} fmt::fmt-header-only)
target_include_directories(bt_tools PRIVATE ${LIBBT_INCLUDE_DIR} pybind11::pybind11 ${Python_INCLUDE_DIRS})


set(LIBBT_PS_DIR "${LIBBT_SOURCE_DIR}/PhaseSpace")
set(LIBBT_PS_SOURCES)


set(LIBBT_WF_DIR "${LIBBT_SOURCE_DIR}/Wavefunctions")
set(LIBBT_WF_SOURCES 
  "${LIBBT_WF_DIR}/Dirac.cpp" 
  "${LIBBT_WF_DIR}/Scalar.cpp"
  "${LIBBT_WF_DIR}/Vector.cpp"
)

add_library(bt_wavefunctions OBJECT ${LIBBT_WF_SOURCES})
target_link_libraries(bt_wavefunctions PUBLIC project_options)
target_include_directories(bt_wavefunctions PRIVATE ${LIBBT_INCLUDE_DIR} pybind11::pybind11 ${Python_INCLUDE_DIRS})

set(LIBBT_AMP_DIR "${LIBBT_SOURCE_DIR}/Amplitudes")
set(LIBBT_AMP_SOURCES
  "${LIBBT_AMP_DIR}/Amplitudes.cpp" 
  "${LIBBT_AMP_DIR}/Currents.cpp"
  "${LIBBT_AMP_DIR}/Propagators.cpp" 
  "${LIBBT_AMP_DIR}/Loop.cpp"
  "${LIBBT_AMP_DIR}/CollierInterface.f90"
)

add_library(bt_amplitudes OBJECT ${LIBBT_AMP_SOURCES})
target_link_libraries(bt_amplitudes PUBLIC project_options PRIVATE collier)
target_include_directories(bt_amplitudes PRIVATE 
  ${LIBBT_INCLUDE_DIR} 
  pybind11::pybind11 
  ${Python_INCLUDE_DIRS}
  "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/collier/modules"
)

set(LIBBT_SPECTRA_DIR "${LIBBT_SOURCE_DIR}/Spectra")
set(LIBBT_SPECTRA_SOURCES
  "${LIBBT_SPECTRA_DIR}/Boost.cpp" 
  "${LIBBT_SPECTRA_DIR}/ChargedKaon.cpp"
  "${LIBBT_SPECTRA_DIR}/ChargedPion.cpp" 
  "${LIBBT_SPECTRA_DIR}/Muon.cpp"
  "${LIBBT_SPECTRA_DIR}/NeutralPion.cpp"
  "${LIBBT_SPECTRA_DIR}/Splitting.cpp"
)

add_library(bt_spectra OBJECT ${LIBBT_SPECTRA_SOURCES})
target_link_libraries(bt_spectra PUBLIC project_options)
target_include_directories(bt_spectra PRIVATE ${LIBBT_INCLUDE_DIR} pybind11::pybind11 ${Python_INCLUDE_DIRS})

set(LIBBT_MODELS_DIR "${LIBBT_SOURCE_DIR}/Models")
set(
  LIBBT_MODELS_SOURCES
  "${LIBBT_MODELS_DIR}/Utils.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoGeneric/FeynmanRules.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoGeneric/SquaredMatrixElements.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoGeneric/Widths.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoGeV/Spectra.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoGeV/Widths.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoGeV/SquaredMatrixElements.cpp"
  # "${LIBBT_MODELS_DIR}/RhNeutrinoMeV/Spectra/TwoBody.cpp"
  # "${LIBBT_MODELS_DIR}/RhNeutrinoMeV/Spectra/VLL.cpp"
  # "${LIBBT_MODELS_DIR}/RhNeutrinoMeV/Spectra/VPiPi.cpp"
  # "${LIBBT_MODELS_DIR}/RhNeutrinoMeV/Spectra/LPiPi0.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoMeV/Spectra.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoMeV/Widths.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoMeV/FeynmanRules.cpp"
  "${LIBBT_MODELS_DIR}/RhNeutrinoMeV/SquaredMatrixElements.cpp"
  "${LIBBT_MODELS_DIR}/StandardModel/FeynmanRules.cpp"
)

add_library(bt_models OBJECT ${LIBBT_MODELS_SOURCES})
target_link_libraries(bt_models PUBLIC project_options)
target_include_directories(bt_models PRIVATE ${LIBBT_INCLUDE_DIR} pybind11::pybind11 ${Python_INCLUDE_DIRS})

# set(
#   LIBBT_SOURCES
#   ${LIBBT_TOOLS_SOURCES}
#   ${LIBBT_PS_SOURCES}
#   ${LIBBT_COLLIER_SOURCES}
#   ${LIBBT_SPECTRA_SOURCES}
#   ${LIBBT_WF_SOURCES}
#   ${LIBBT_AMP_SOURCES}
#   ${LIBBT_MODELS_SOURCES}
# )


set(LIBBT bt)
add_library(${LIBBT} STATIC 
  $<TARGET_OBJECTS:bt_tools>
  $<TARGET_OBJECTS:bt_wavefunctions>
  $<TARGET_OBJECTS:bt_amplitudes>
  $<TARGET_OBJECTS:bt_spectra>
  $<TARGET_OBJECTS:bt_models>
)
target_link_libraries(
  ${LIBBT}
  PUBLIC project_options
  PRIVATE fmt::fmt-header-only
  Threads::Threads
  GSL::gsl
  GSL::gslcblas
  collier
  tbb
  ${Python_LIBRARIES}
  ${PYTHIA8_LIB}
  ${CUBA_LIB}
)
target_include_directories(
  ${LIBBT} PUBLIC ${LIBBT_INCLUDE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/collier/modules"
  "/usr/local/include"
  pybind11::pybind11
  ${Python_INCLUDE_DIRS}
)

# ============================================================================
# ---- Build testing utils ---------------------------------------------------
# ============================================================================

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

set(LIBTESTUTILS bttestutils)
set(TESTUTILS_SOURCES "${TEST_DIR}/Tools.cpp")
add_library(${LIBTESTUTILS} ${TESTUTILS_SOURCES})
target_link_libraries(${LIBTESTUTILS} PRIVATE ${LIBBT} helas fmt::fmt-header-only)
target_include_directories(${LIBTESTUTILS} PUBLIC ${LIBBT_INCLUDE_DIR})

# ============================================================================
# ---- Build testing files ---------------------------------------------------
# ============================================================================

find_package(GTest CONFIG REQUIRED)
find_package(Catch2 REQUIRED)

set(
  TEST_FILES "TestPhaseSpace" "TestDiracWf" "TestVectorWf" "TestAmplitudes"
  "TestScratch" "TestWavefunctionsToHelas" "TestSpectraRambo" "TestDecaySpectum"
  "TestMuonDecay"
)
set(LIB_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

foreach(tfile ${TEST_FILES})
  add_executable(
    ${tfile} "${LIB_TEST_DIR}/${tfile}.cpp"
    "${LIB_TEST_DIR}/Tools.cpp"
  )
  target_link_libraries(
    ${tfile}
    PRIVATE ${LIBBT}
    ${LIBTESTUTILS}
    helas
    project_options
    # project_warnings
    Threads::Threads
    ${PYTHIA8_LIB}
    GTest::gtest_main
    Catch2::Catch2
    fmt::fmt-header-only
  )
  target_include_directories(${tfile} PUBLIC ${LIBBT_INCLUDE_DIR})
endforeach()

set(RHN_TEST_FILES "TestWidthsGeV" "TestWidthsMeV")
set(RHN_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/RhNeutrino")

foreach(tfile ${RHN_TEST_FILES})
  add_executable(
    ${tfile} "${RHN_TEST_DIR}/${tfile}.cpp"
    "${LIB_TEST_DIR}/Tools.cpp"
  )
  target_link_libraries(
    ${tfile}
    PRIVATE ${LIBBT}
    ${LIBTESTUTILS}
    helas
    project_options
    # project_warnings
    Threads::Threads
    GTest::gtest_main
    Catch2::Catch2
    fmt::fmt-header-only
  )
  target_include_directories(${tfile} PUBLIC ${LIBBT_INCLUDE_DIR})
endforeach()
