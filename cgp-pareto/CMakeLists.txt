cmake_minimum_required(VERSION 3.14)
project(cgp_hes CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimise for speed (evolutionary runs are compute-heavy)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(cgp_hes
    src/main.cpp
)

target_include_directories(cgp_hes PRIVATE include)

# Enable warnings
target_compile_options(cgp_hes PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
