cmake_minimum_required(VERSION 3.9)
#set(CMAKE_BUILD_TYPE Debug)
include_directories(${PROJECT_SOURCE_DIR}/headers)

add_executable(BanditPAM main.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(BanditPAM PUBLIC OpenMP::OpenMP_CXX)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(BanditPAM PUBLIC ${ARMADILLO_LIBRARIES})

add_library(BanditPAM_LIB kmedoids_ucb.cpp)
target_link_libraries(BanditPAM_LIB PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)

