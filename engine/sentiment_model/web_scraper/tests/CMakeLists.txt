# ============================================================================
# Test Suite CMakeLists.txt for ML Trading Bot Web Scraper
# ============================================================================
# This file configures the test suite for the web scraper components.
# Uses Google Test framework for unit testing.
# ============================================================================

# Test executable for scraper functionality
add_executable(
    scraper_tests
    test_scraper.cpp
    test_main_functions.cpp
    test_runner.cpp
    main_functions_for_testing.cpp
)

# Link test executable with Google Test and our scraper library
target_link_libraries(
    scraper_tests
    scraper_lib          # Our scraper library
    gtest_main           # Google Test main function
    gtest                # Google Test framework
    gmock                # Google Mock (useful for future mocking needs)
    CURL::libcurl        # Direct curl dependency for integration tests
    fmt::fmt             # fmt library dependency
)

# Include directories so tests can access scraper headers
target_include_directories(
    scraper_tests 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(scraper_tests)

# Add custom target to run tests with verbose output
add_custom_target(
    run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS scraper_tests
    COMMENT "Running all tests with verbose output"
)