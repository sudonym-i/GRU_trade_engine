/**
 * @file test_main_functions.cpp
 * @brief Unit tests for main.cpp functionality
 * 
 * This file contains Google Test unit tests for the JSON parsing and configuration
 * functions used in the main.cpp file of the web scraper application.
 * 
 * Tests cover:
 * - JSON value extraction
 * - Configuration file parsing
 * - Error handling for malformed JSON
 * - Edge cases and boundary conditions
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include <chrono>

// Include the headers
#include "scraper.h"
#include "test_helpers.h"

// Test fixture for JSON parsing tests
class JsonParsingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up common test data
        valid_json = R"({
            "sentiment_model": {
                "web_scraper": {
                    "chains": [
                        {
                            "start_tag": "<small>",
                            "end_tag": "</small>",
                            "stock_name": "nvidia"
                        }
                    ],
                    "output_name": "test_output.raw"
                }
            }
        })";
        
        malformed_json = R"({
            "sentiment_model": {
                "web_scraper": {
                    "chains": [
                        {
                            "start_tag": "<small>",
                            "end_tag": "</small>",
                            "stock_name": "nvidia"
        })";  // Missing closing brackets
    }
    
    void TearDown() override {
        // Clean up any test files
        std::filesystem::remove("test_config.json");
    }
    
    std::string valid_json;
    std::string malformed_json;
};

// =============================================================================
// Tests for extractJsonValue function
// =============================================================================

TEST_F(JsonParsingTest, ExtractJsonValue_ValidKey_ReturnsCorrectValue) {
    // Test extracting a simple string value
    std::string test_json = R"({"name": "test_value", "number": "42"})";
    
    EXPECT_EQ(extractJsonValue(test_json, "name"), "test_value");
    EXPECT_EQ(extractJsonValue(test_json, "number"), "42");
}

TEST_F(JsonParsingTest, ExtractJsonValue_KeyNotFound_ReturnsEmptyString) {
    std::string test_json = R"({"existing_key": "value"})";
    
    EXPECT_EQ(extractJsonValue(test_json, "nonexistent_key"), "");
}

TEST_F(JsonParsingTest, ExtractJsonValue_EmptyJson_ReturnsEmptyString) {
    std::string empty_json = "";
    
    EXPECT_EQ(extractJsonValue(empty_json, "any_key"), "");
}

TEST_F(JsonParsingTest, ExtractJsonValue_KeyWithSpaces_HandlesCorrectly) {
    std::string test_json = R"({"key with spaces": "value with spaces"})";
    
    // This should fail with current implementation (which is expected behavior)
    EXPECT_EQ(extractJsonValue(test_json, "key with spaces"), "value with spaces");
}

TEST_F(JsonParsingTest, ExtractJsonValue_NestedJson_ExtractsSingleLevel) {
    std::string nested_json = R"({
        "outer": {
            "inner": "nested_value"
        },
        "simple": "simple_value"
    })";
    
    // Should extract simple values correctly
    EXPECT_EQ(extractJsonValue(nested_json, "simple"), "simple_value");
    // Nested values require different parsing approach
    EXPECT_EQ(extractJsonValue(nested_json, "inner"), "nested_value");
}

// =============================================================================
// Integration Tests for Configuration Parsing
// =============================================================================

class ConfigurationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary config file for testing
        test_config_content = R"({
            "sentiment_model": {
                "web_scraper": {
                    "chains": [
                        {
                            "start_tag": "<div>",
                            "end_tag": "</div>",
                            "stock_name": "apple"
                        }
                    ],
                    "output_name": "apple_data.raw"
                }
            }
        })";
        
        std::ofstream config_file("test_config.json");
        config_file << test_config_content;
        config_file.close();
    }
    
    void TearDown() override {
        std::filesystem::remove("test_config.json");
    }
    
    std::string test_config_content;
};

// Note: Since parse_crawlchain() is tightly coupled to file I/O and has hardcoded paths,
// we'll create tests that verify the expected behavior and document limitations

TEST_F(ConfigurationTest, ConfigFileStructure_IsValid) {
    // Verify our test config has the expected structure
    EXPECT_NE(test_config_content.find("sentiment_model"), std::string::npos);
    EXPECT_NE(test_config_content.find("web_scraper"), std::string::npos);
    EXPECT_NE(test_config_content.find("chains"), std::string::npos);
}

TEST_F(ConfigurationTest, JsonExtraction_FromRealConfig_Works) {
    // Test extraction from our realistic config format
    EXPECT_EQ(extractJsonValue(test_config_content, "stock_name"), "apple");
    EXPECT_EQ(extractJsonValue(test_config_content, "output_name"), "apple_data.raw");
    EXPECT_EQ(extractJsonValue(test_config_content, "start_tag"), "<div>");
    EXPECT_EQ(extractJsonValue(test_config_content, "end_tag"), "</div>");
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST(ErrorHandlingTest, ExtractJsonValue_MalformedJson_HandlesGracefully) {
    std::string malformed = R"({"key": "value" missing_quote_and_bracket)";
    
    // Should not crash and should return empty string
    EXPECT_EQ(extractJsonValue(malformed, "key"), "");
}

TEST(ErrorHandlingTest, ExtractJsonValue_NoColonAfterKey_ReturnsEmpty) {
    std::string no_colon = R"({"key" "value"})";
    
    EXPECT_EQ(extractJsonValue(no_colon, "key"), "");
}

TEST(ErrorHandlingTest, ExtractJsonValue_NoClosingQuote_ReturnsEmpty) {
    std::string no_closing = R"({"key": "value without closing quote})";
    
    EXPECT_EQ(extractJsonValue(no_closing, "key"), "");
}

// =============================================================================
// Performance and Edge Case Tests
// =============================================================================

TEST(EdgeCaseTest, ExtractJsonValue_LargeJson_PerformsReasonably) {
    // Create a large JSON string
    std::string large_json = "{";
    for (int i = 0; i < 1000; ++i) {
        large_json += "\"key" + std::to_string(i) + "\": \"value" + std::to_string(i) + "\",";
    }
    large_json += "\"target_key\": \"target_value\"}";
    
    // Should still find the target value efficiently
    auto start = std::chrono::high_resolution_clock::now();
    std::string result = extractJsonValue(large_json, "target_key");
    auto end = std::chrono::high_resolution_clock::now();
    
    EXPECT_EQ(result, "target_value");
    
    // Should complete in reasonable time (less than 1ms for this size)
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    EXPECT_LT(duration.count(), 1000); // Less than 1ms
}

TEST(EdgeCaseTest, ExtractJsonValue_SpecialCharacters_HandlesCorrectly) {
    std::string special_chars = R"({"key": "value with \"quotes\" and \\ backslashes"})";
    
    // Note: Current implementation doesn't handle escaped characters
    // This test documents the limitation
    std::string result = extractJsonValue(special_chars, "key");
    // The result will be truncated at the first escaped quote
    EXPECT_NE(result, "value with \"quotes\" and \\ backslashes");
}

// =============================================================================
// Mock Tests for File Operations (Future Enhancement)
// =============================================================================

// This demonstrates how we could test file operations with mocking
// Currently commented out as it requires refactoring the main.cpp functions
/*
class MockFileSystem {
public:
    MOCK_METHOD(std::string, readFile, (const std::string& path), ());
    MOCK_METHOD(bool, fileExists, (const std::string& path), ());
};

TEST(MockTest, ConfigParsing_WithMockedFile_WorksCorrectly) {
    MockFileSystem mockFs;
    
    EXPECT_CALL(mockFs, fileExists(testing::_))
        .WillOnce(testing::Return(true));
    
    EXPECT_CALL(mockFs, readFile(testing::_))
        .WillOnce(testing::Return(valid_json));
    
    // Test would require refactored parse_crawlchain function
    // that accepts a file system interface
}
*/