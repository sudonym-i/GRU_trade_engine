/**
 * @file scraper.h
 * @brief Header file for web scraping functionality used in ML trading bot sentiment analysis
 * 
 * This header defines the DataList class and associated structures for web content extraction.
 * The scraper is designed to fetch HTML content from URLs and extract text between specified
 * HTML tags for sentiment analysis data collection.
 * 
 * Key Features:
 * - HTTP request handling using libCurl
 * - HTML content parsing with configurable tag boundaries
 * - Modern C++ practices with smart pointers and containers
 * - Memory-safe operations with RAII
 * 
 * @author ML Trading Bot Project
 * @version 2.0
 * @date 2024
 */

#ifndef SCRAPER_H
#define SCRAPER_H

#include <string>      // Required for std::string operations
#include <vector>      // Required for std::vector containers
#include <memory>      // Required for smart pointers
#include <iosfwd>      // Forward declarations for iostream types

/**
 * @brief Configuration structure for a single scraping operation
 */
struct ScrapeConfig {
    std::string url;       ///< Target URL to scrape
    std::string start_tag; ///< HTML start tag for content extraction
    std::string end_tag;   ///< HTML end tag for content extraction
};

/**
 * @brief Character used to separate different pieces of scraped content in output
 * 
 * This separator is inserted between different content blocks extracted from
 * the same or different web pages to maintain data structure in output files.
 */
const char OUTPUT_SEPARATOR = '\n';

/**
 * @class DataList
 * @brief Main class for managing web scraping operations and data storage
 * 
 * DataList implements a linked list data structure to store and manage scraped
 * web content. Each node contains extracted text from a single URL, parsed according
 * to specified HTML tag boundaries. The class handles:
 * 
 * - HTTP requests using libCurl
 * - HTML content parsing and extraction
 * - Memory management with smart pointers
 * - Output formatting for sentiment analysis data
 * 
 * The class uses modern C++ practices including smart pointers for automatic
 * memory management and provides methods for writing all collected data to output streams.
 */
class DataList {
public:
    /**
     * @brief Destructor - automatically cleans up all allocated memory
     * 
     * Uses smart pointers for automatic memory management. The compiler-generated
     * destructor is sufficient as std::unique_ptr handles cleanup automatically.
     * 
     * @note Smart pointers eliminate the need for manual memory management
     * @note No explicit cleanup code needed - RAII handles everything
     */
    ~DataList() = default;

    /**
     * @brief Constructor that performs all scraping operations during initialization
     * 
     * Creates a DataList object and immediately performs web scraping for all
     * provided configurations. For each configuration, it:
     * 1. Performs an HTTP GET request
     * 2. Extracts content between the specified start and end tags
     * 3. Stores the results in a new linked list node
     * 
     * @param configs Vector of ScrapeConfig objects containing URLs and tag boundaries
     * 
     * @note Constructor performs all HTTP operations - may take significant time
     * @note Failed scraping operations store empty strings in the corresponding node
     * @note Uses modern C++ containers for better memory management
     */
    explicit DataList(const std::vector<ScrapeConfig>& configs);

    /**
     * @brief Writes all scraped content to the provided output stream
     * 
     * This method traverses the entire linked list and outputs all stored content
     * to the specified stream. It handles empty data gracefully (from failed
     * scraping operations) and maintains the original content formatting.
     * 
     * @param out Reference to output stream (file, console, stringstream, etc.)
     * 
     * @note Does not add additional formatting - preserves original content structure
     * @note Safely handles nodes with empty data (skips them)
     * @note Can be called multiple times on the same DataList object
     */
    void write(std::ostream& out);

    /**
     * @brief libCurl callback function for receiving HTTP response data
     * 
     * Static callback function used by libCurl to handle incoming HTTP data.
     * Appends received data to the provided string buffer. Must be static
     * to comply with libCurl callback function signature requirements.
     * 
     * @param contents Pointer to received data buffer
     * @param size Size of each data element
     * @param nmemb Number of data elements
     * @param output Pointer to std::string where data should be appended
     * @return size_t Number of bytes processed (size * nmemb for success)
     * 
     * @note Required for libCurl integration - do not modify signature
     * @note Returns the number of bytes processed for libCurl status tracking
     * @note Made public for testing purposes
     */
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output);

private:
    /**
     * @struct Node
     * @brief Individual node in the linked list storing scraped content
     * 
     * Each node represents the result of scraping a single URL and contains:
     * - The scraped content as a string
     * - A smart pointer to the next node in the list
     * 
     * @note Uses smart pointers for automatic memory management
     * @note Empty data string indicates scraping failure for that URL
     */
    struct Node {
        std::string data;                    ///< Scraped content
        std::unique_ptr<Node> next = nullptr; ///< Smart pointer to next node
    };

    /**
     * @brief Smart pointer to the first node in the linked list
     * 
     * Head pointer for traversing the linked list of scraped content.
     * nullptr indicates an empty list. Uses smart pointer for automatic cleanup.
     */
    std::unique_ptr<Node> head = nullptr;

    /**
     * @brief Modern string comparison function for HTML tag matching
     * 
     * This method searches for a target tag string starting at a given position
     * in HTML content using modern C++ string operations for better safety and
     * performance compared to character-by-character comparison.
     * 
     * @param html_content The HTML content to search in
     * @param pos Starting position to search from (modified to position after match if found)
     * @param target Target tag string to search for
     * @return bool True if tag found at position, false otherwise
     * 
     * @note Modifies 'pos' parameter: advances past match on success
     * @note Uses std::string operations for better safety and performance
     */
    bool findTag(const std::string& html_content, size_t& pos, const std::string& target);


    /**
     * @brief Performs HTTP request and content extraction for a single URL
     * 
     * Core scraping function that:
     * 1. Initializes libCurl for HTTP operations
     * 2. Performs GET request to the specified URL
     * 3. Processes the received HTML content
     * 4. Extracts content between specified start and end tags
     * 5. Returns extracted content string
     * 
     * @param config ScrapeConfig containing URL and tag boundaries
     * @return std::string Extracted content, or empty string on failure
     * 
     * @note Uses value semantics instead of raw pointers for better safety
     * @note Performs console output showing scraping progress and results
     * @note Handles libCurl errors and network failures gracefully
     */
    std::string scrape(const ScrapeConfig& config);

    /**
     * @brief Parses HTML content to extract text between specified tags
     * 
     * Searches through HTML content and extracts all text that appears between
     * the specified start and end tag markers. Handles multiple occurrences
     * of the tag pair within the same HTML document.
     * 
     * @param html_content The full HTML content string
     * @param start_tag HTML start tag marking beginning of desired content
     * @param end_tag HTML end tag marking end of desired content
     * @return std::string String containing all extracted content
     * 
     * @note Uses value semantics for better memory safety
     * @note Includes OUTPUT_SEPARATOR between different content blocks
     * @note Uses std::string operations for better safety and readability
     * @note Handles malformed HTML gracefully without crashing
     */
    std::string extractContent(const std::string& html_content, 
                              const std::string& start_tag, 
                              const std::string& end_tag);
};

#endif // SCRAPER_H