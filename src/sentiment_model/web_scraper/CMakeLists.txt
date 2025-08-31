# ============================================================================
# CMakeLists.txt - Build Configuration for ML Trading Bot Web Scraper
# ============================================================================
# This CMake configuration file defines the build process for the web scraper
# component used in sentiment analysis data collection for the ML trading bot.
# 
# Project Structure:
# - Builds a C++ executable (webscrape.exe) from main.cpp and scraper.cpp
# - Links against libCurl library for HTTP request functionality
# - Designed for cross-platform compatibility (Linux, Windows, macOS)
# 
# Dependencies:
# - CMake 3.10 or higher
# - libCurl development library
# - C++11 compatible compiler
# 
# Build Instructions:
#   mkdir build && cd build
#   cmake ..
#   make
# 
# Author: ML Trading Bot Project
# Version: 1.0
# Date: 2024
# ============================================================================

# Set the minimum CMake version required for this project
# CMake 3.10 provides good cross-platform support and modern C++ features
cmake_minimum_required(VERSION 3.10)

# Define the project name and enable C++ language support
# The executable will be named 'webscrape.exe' for consistency across platforms
project(webscrape.exe)

# Create executable target from source files
# Links together the main application logic (main.cpp) and scraping functionality (scraper.cpp)
add_executable(webscrape.exe main.cpp scraper.cpp)

# Locate the libCurl library on the system
# REQUIRED flag ensures build fails if libCurl is not found
# libCurl provides HTTP/HTTPS client functionality for web scraping
find_package(CURL REQUIRED)
find_package(fmt REQUIRED)

# Link the libCurl library to the executable
# Uses modern CMake target-based linking for proper include paths and library flags
# CURL::libcurl is the imported target provided by FindCURL module
target_link_libraries(webscrape.exe CURL::libcurl fmt::fmt)

# ============================================================================
# Google Test Configuration for Unit Testing
# ============================================================================

# Include CTest for running tests
include(CTest)
enable_testing()

# Only build tests if this is the main project (not included as subdirectory)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Fetch Google Test from GitHub
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0  # Use latest stable version
    )
    
    # Configure Google Test
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Create a library from scraper sources (for testing)
    # This allows tests to link against scraper functionality without duplicating code
    add_library(scraper_lib scraper.cpp)
    target_link_libraries(scraper_lib CURL::libcurl fmt::fmt)
    
    # Add the tests subdirectory
    add_subdirectory(./tests)
endif()
