/**
 * @file test_runner.cpp
 * @brief Test runner and setup for Google Test suite
 * 
 * This file provides the main entry point for the Google Test suite and any
 * global test setup/teardown that might be needed for the web scraper tests.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>

/**
 * @brief Global test environment setup
 * 
 * This class handles any global setup/teardown that needs to happen
 * before/after all tests run.
 */
class WebScraperTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        std::cout << "Setting up global test environment..." << std::endl;
        
        // Set environment variables for testing
        // This helps skip network tests in CI environments
        if (!std::getenv("SKIP_NETWORK_TESTS")) {
            std::cout << "Network tests enabled. Set SKIP_NETWORK_TESTS=1 to disable." << std::endl;
        } else {
            std::cout << "Network tests disabled by environment variable." << std::endl;
        }
        
        // Initialize any global resources if needed
        std::cout << "Global test environment ready." << std::endl;
    }
    
    void TearDown() override {
        std::cout << "Cleaning up global test environment..." << std::endl;
        // Clean up any global resources
    }
};

/**
 * @brief Custom test listener for additional output formatting
 */
class WebScraperTestListener : public ::testing::EmptyTestEventListener {
public:
    void OnTestStart(const ::testing::TestInfo& test_info) override {
        std::cout << "\n[RUNNING] " << test_info.test_suite_name() 
                  << "." << test_info.name() << std::endl;
    }
    
    void OnTestEnd(const ::testing::TestInfo& test_info) override {
        if (test_info.result()->Passed()) {
            std::cout << "[  PASS  ] " << test_info.test_suite_name() 
                      << "." << test_info.name() << std::endl;
        } else {
            std::cout << "[  FAIL  ] " << test_info.test_suite_name() 
                      << "." << test_info.name() << std::endl;
        }
    }
};

/**
 * @brief Main function for the test suite
 * 
 * Initializes Google Test and runs all tests with custom configuration.
 */
int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Add global test environment
    ::testing::AddGlobalTestEnvironment(new WebScraperTestEnvironment);
    
    // Add custom test listener for better output formatting
    ::testing::TestEventListeners& listeners = 
        ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new WebScraperTestListener);
    
    // Print test configuration
    std::cout << "\n=== Web Scraper Test Suite ===" << std::endl;
    std::cout << "Google Test Version: " << "1.14.0" << std::endl;
    std::cout << "Test executable: " << argv[0] << std::endl;
    
    if (argc > 1) {
        std::cout << "Command line arguments: ";
        for (int i = 1; i < argc; ++i) {
            std::cout << argv[i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nRunning tests..." << std::endl;
    
    // Run all tests
    int result = RUN_ALL_TESTS();
    
    // Print summary
    if (result == 0) {
        std::cout << "\n✅ All tests passed!" << std::endl;
    } else {
        std::cout << "\n❌ Some tests failed. Check output above." << std::endl;
    }
    
    return result;
}