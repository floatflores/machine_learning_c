cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(Machine_Learning C)

add_compile_options(-Wall -Wextra -g)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 指定 C 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(Machine_Learning
    src/main.c
    src/arena.c
    src/prng.c
)

target_link_libraries(Machine_Learning m)

# 添加头文件搜索路径
target_include_directories(Machine_Learning
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
