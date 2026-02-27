cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(Machine_Learning C)

add_compile_options(-Wall -Wextra)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(Machine_Learning
    src/main.c
    src/arena.c
    src/prng.c
    src/matrix.c
    src/dataset.c
    src/model.c
)

target_link_libraries(Machine_Learning m)

target_include_directories(Machine_Learning
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
