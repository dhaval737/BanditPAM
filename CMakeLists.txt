cmake_minimum_required(VERSION 3.17)
project(BanditPAM VERSION 1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

list(APPEND CMAKE_MODULE_PATH /usr/share/cmake/Modules/)

add_subdirectory(src)
