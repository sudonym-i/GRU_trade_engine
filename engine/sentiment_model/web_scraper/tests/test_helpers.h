/**
 * @file test_helpers.h
 * @brief Helper functions and declarations for testing
 * 
 * This header provides forward declarations and helper functions needed
 * for testing the web scraper components.
 */

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <string>
#include <chrono>

// Forward declarations for functions from main.cpp that we want to test
// Note: In a production setup, these would be in a separate header file
std::string extractJsonValue(const std::string& json, const std::string& key);

// Helper macros for timing tests
#define START_TIMER() auto start_time = std::chrono::high_resolution_clock::now()
#define END_TIMER_MS() \
    std::chrono::duration_cast<std::chrono::milliseconds>( \
        std::chrono::high_resolution_clock::now() - start_time).count()

#define END_TIMER_US() \
    std::chrono::duration_cast<std::chrono::microseconds>( \
        std::chrono::high_resolution_clock::now() - start_time).count()

#endif // TEST_HELPERS_H