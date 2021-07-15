include_directories(/opt/intel/oneapi/intelpython/python3.7/include/python3.7m)

include_directories(${PROJECT_SOURCE_DIR}/headers/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/headers/pybind11/include/pybind11)
include_directories(${PROJECT_SOURCE_DIR}/headers/pybind11)
include_directories(/usr/local/include/pybind11/detail)
include_directories(/usr/local/include/pybind11)

include_directories(/Users/motiwari/.local/lib/python3.7/site-packages/numpy/core/include)
include_directories(/Users/motiwari/.local/lib/python3.7/site-packages/pybind11/include)

include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include/carma_bits)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include)


include_directories(${PROJECT_SOURCE_DIR}/headers)

add_executable(BanditPAM main.cpp)

add_library(BanditPAM_LIB kmedoids_ucb.cpp)
target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)

find_package(OpenMP REQUIRED)
target_link_libraries(BanditPAM PUBLIC OpenMP::OpenMP_CXX)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(BanditPAM PUBLIC ${ARMADILLO_LIBRARIES})
