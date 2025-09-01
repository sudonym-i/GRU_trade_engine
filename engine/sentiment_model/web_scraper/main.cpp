
/**
 * @file main.cpp
 * @brief Entry point for the web scraper application that extracts content from websites
 *        for sentiment analysis data collection as part of an ML trading bot system.
 * 
 * This application reads scraping configuration from a JSON file, performs HTTP requests
 * to target URLs, and extracts content between specified HTML tags. The scraped data
 * is then saved to output files for further processing in sentiment analysis models.
 * 
 * @author ML Trading Bot Project
 * @version 1.0
 * @date 2024
 */

#include <string>      // For std::string operations
#include <iostream>    // For console input/output operations
#include <fstream>     // For file input/output operations
#include <sstream>     // For string stream operations
#include <vector>      // For std::vector containers
#include <stdexcept>   // For exception handling
#include "scraper.h"   // Custom header for DataList class and scraping functionality
#include <fmt/format.h>  // For fmt::format library

/**
 * @brief File path to the JSON configuration file containing scraping instructions
 * 
 * This constant defines the location of the chain.json file which contains:
 * - Target URLs to scrape
 * - HTML start and end tags that define content boundaries
 * - Output file naming configuration
 * 
 * The file uses JSON format for structured configuration data.
 */
const std::string CHAIN_PATH = "../../../../config.json";

/**
 * @brief Complete configuration parsed from JSON file
 */
struct ChainConfig {
    std::string output_name;                ///< Output filename for scraped data
    std::vector<ScrapeConfig> chains;      ///< Vector of scraping configurations
};

/**
 * @brief Parses the JSON configuration file to extract scraping parameters
 * 
 * This function reads the chain.json file and extracts the necessary information
 * for web scraping operations including URLs, HTML tag boundaries, and output
 * file configuration. It handles JSON parsing manually using string operations.
 * 
 * @return ChainConfig Parsed configuration containing output filename and scraping chains
 * @throws std::runtime_error If file cannot be opened or JSON parsing fails
 * 
 * @note Currently supports parsing a single chain entry from the JSON array
 * @note Uses modern C++ containers for better memory management
 */

ChainConfig parse_crawlchain();

/**
 * @brief Main entry point for the web scraper application
 * 
 * This function orchestrates the entire web scraping process:
 * 1. Parses the JSON configuration file to extract scraping parameters
 * 2. Creates a DataList object that performs HTTP requests and content extraction
 * 3. Writes the scraped and parsed content to the specified output file
 * 
 * The application uses modern C++ practices with RAII for resource management
 * and exception handling for error conditions. All scraped content is
 * automatically filtered to extract only the text between specified HTML tags.
 * 
 * @return int Exit status (0 for success, 1 for failure)
 * 
 * @note Uses exception handling for robust error management
 * @note Employs RAII for automatic resource cleanup
 * @note Output files are created in the current working directory
 */
int main() {
    try {
        // Parse the JSON configuration file to get scraping parameters
        ChainConfig config = parse_crawlchain();
        
        // Create DataList object which automatically performs all HTTP requests
        // and content extraction based on the parsed configuration
        DataList crawledData(config.chains);

        // Open output file for writing scraped content (RAII ensures cleanup)
        std::ofstream outfile("../../raw_data/" + config.output_name);
        if (!outfile.is_open()) {
            throw std::runtime_error("Failed to open output file: " + config.output_name);
        }

        // Write all scraped and processed content to the output file
        crawledData.write(outfile);
        
        std::cout << "Successfully wrote scraped data to: " << config.output_name << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} // END main


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

/**
 * @brief Parses the chain.json configuration file and extracts scraping parameters
 * 
 * This function implements the core configuration parsing logic for the web scraper.
 * It reads the JSON configuration file and extracts all necessary parameters for
 * web scraping operations. The function handles both global configuration (output
 * filename) and individual chain configurations (URLs and HTML tag boundaries).
 * 
 * Expected JSON structure:
 * {
 *   "output_name": "filename.csv",
 *   "chains": [
 *     {
 *       "start_tag": "<div class='content'>",
 *       "end_tag": "</div>",
 *       "url": "https://example.com"
 *     }
 *   ]
 * }
 * 
 * @return ChainConfig Parsed configuration with output filename and scraping chains
 * @throws std::runtime_error If file cannot be opened or JSON parsing fails
 * 
 * @note Currently supports parsing a single chain entry from the JSON array
 * @note Uses modern C++ containers and exception handling
 * @note Provides better error reporting through exceptions
 */
ChainConfig parse_crawlchain() {
    // Open the JSON configuration file
    std::ifstream crawlchain(CHAIN_PATH);

    // Validate file access
    if (!crawlchain.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + CHAIN_PATH);
    }

    // Read entire file content into memory for parsing
    std::stringstream buffer;
    buffer << crawlchain.rdbuf();
    std::string jsonContent = buffer.str();

    ChainConfig config;
    
    // Navigate to the sentiment_model.web_scraper section
    size_t sentimentPos = jsonContent.find("\"sentiment_model\"");
    if (sentimentPos == std::string::npos) {
        throw std::runtime_error("No 'sentiment_model' section found in JSON configuration");
    }
    
    size_t webScraperPos = jsonContent.find("\"web_scraper\"", sentimentPos);
    if (webScraperPos == std::string::npos) {
        throw std::runtime_error("No 'web_scraper' section found in sentiment_model configuration");
    }
    
    // Find the web_scraper object boundaries
    size_t webScraperStart = jsonContent.find("{", webScraperPos);
    size_t webScraperEnd = jsonContent.find("}\n  }", webScraperStart);
    
    if (webScraperStart == std::string::npos || webScraperEnd == std::string::npos) {
        throw std::runtime_error("Invalid JSON structure in web_scraper section");
    }
    
    std::string webScraperSection = jsonContent.substr(webScraperStart, webScraperEnd - webScraperStart + 1);
    
    // Extract output filename from the web_scraper section
    config.output_name = extractJsonValue(webScraperSection, "output_name");
    if (config.output_name.empty()) {
        throw std::runtime_error("Missing or empty 'output_name' in web_scraper configuration");
    }

    // Locate the chains array in the web_scraper section
    size_t chainsPos = webScraperSection.find("\"chains\"");
    if (chainsPos == std::string::npos) {
        throw std::runtime_error("No 'chains' array found in web_scraper configuration");
    }

    // Parse the first (and currently only supported) chain entry
    // Find the array opening bracket
    size_t arrayStart = webScraperSection.find("[", chainsPos);
    // Find the first object in the array
    size_t objectStart = webScraperSection.find("{", arrayStart);
    size_t objectEnd = webScraperSection.find("}", objectStart);
    
    // Validate JSON structure
    if (objectStart == std::string::npos || objectEnd == std::string::npos) {
        throw std::runtime_error("Invalid JSON structure in chains array");
    }

    // Extract the individual chain object for detailed parsing
    std::string chainObject = webScraperSection.substr(objectStart, objectEnd - objectStart + 1);
    
    // Extract individual chain configuration values
    ScrapeConfig scrapeConfig;
    scrapeConfig.start_tag = extractJsonValue(chainObject, "start_tag");
    scrapeConfig.end_tag = extractJsonValue(chainObject, "end_tag");
    
    // Extract stock_name and construct the URL dynamically
    std::string stock_name = extractJsonValue(chainObject, "stock_name");
    if (stock_name.empty()) {
        throw std::runtime_error("Missing required field 'stock_name' in chain configuration");
    }
    
    // Construct the URL using fmt::format with the stock name
    scrapeConfig.url = fmt::format("https://filmot.com/search/{}%20stock/1?gridView=1", stock_name);
    
    // Validate that all required fields are present
    if (scrapeConfig.start_tag.empty() || scrapeConfig.end_tag.empty() || scrapeConfig.url.empty()) {
        throw std::runtime_error("Missing required fields (start_tag, end_tag, or stock_name) in chain configuration");
    }
    
    // Add the parsed configuration to the chains vector
    config.chains.push_back(scrapeConfig);
    
    return config;
}

