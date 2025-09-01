/**
 * @file main_functions_for_testing.cpp
 * @brief Implementation of main.cpp functions for testing
 * 
 * This file contains the implementation of functions from main.cpp that we want to test.
 * In a production setup, these would be refactored into a separate library.
 */

#include <string>

/**
 * @brief Simple JSON value extractor for configuration parsing
 * 
 * This function implements a basic JSON parser that extracts string values
 * associated with specific keys from a JSON string. It performs manual string
 * searching and parsing without using external JSON libraries.
 * 
 * The parsing process:
 * 1. Searches for the key surrounded by quotes
 * 2. Finds the colon separator after the key
 * 3. Locates the opening quote of the value
 * 4. Finds the closing quote of the value
 * 5. Extracts and returns the substring between the value quotes
 * 
 * @param json The JSON string to parse
 * @param key The key name to search for (without quotes)
 * @return std::string The extracted value, or empty string if key not found
 * 
 * @note This is a simplified JSON parser that assumes well-formed JSON
 * @note Does not handle escaped quotes or complex JSON structures
 * @note Returns empty string on parsing errors or missing keys
 */
std::string extractJsonValue(const std::string& json, const std::string& key) {
    // Construct the search pattern with quotes around the key
    std::string searchKey = "\"" + key + "\"";
    size_t keyPos = json.find(searchKey);
    if (keyPos == std::string::npos) return "";
    
    // Find the colon separator after the key
    size_t colonPos = json.find(":", keyPos);
    if (colonPos == std::string::npos) return "";
    
    // Find the opening quote of the value
    size_t startQuote = json.find("\"", colonPos);
    if (startQuote == std::string::npos) return "";
    
    // Find the closing quote of the value
    size_t endQuote = json.find("\"", startQuote + 1);
    if (endQuote == std::string::npos) return "";
    
    // Extract and return the value between the quotes
    return json.substr(startQuote + 1, endQuote - startQuote - 1);
}