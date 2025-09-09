
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

# endif()