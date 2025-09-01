/**
 * @file scraper.cpp
 * @brief Implementation of web scraping functionality for sentiment analysis data collection
 * 
 * This file implements the DataList class methods for performing HTTP requests,
 * parsing HTML content, and extracting text between specified tags. The implementation
 * uses libCurl for HTTP operations and modern C++ practices for safe memory management.
 * 
 * Key Implementation Details:
 * - Uses smart pointers for automatic memory management
 * - Implements modern C++ string operations for HTML parsing
 * - Provides memory-safe operations with RAII principles
 * - Handles HTTP errors and network failures gracefully
 * - Supports console output for monitoring scraping progress
 * 
 * @author ML Trading Bot Project
 * @version 2.0
 * @date 2024
 */

#include <iostream>      // For console input/output operations
#include <string>        // For std::string operations  
#include <curl/curl.h>   // For HTTP request functionality
#include <sstream>       // For string stream operations
#include "scraper.h"     // Header file with class definitions and constants

/**
 * @brief Constructor implementation - performs all web scraping operations during object creation
 * 
 * This constructor immediately initiates scraping operations for all provided configurations.
 * It builds a linked list where each node contains the results of scraping a single
 * URL with its associated HTML tag boundaries.
 * 
 * Construction Process:
 * 1. Create the first node and scrape the first URL
 * 2. For each additional configuration, create a new node and link it to the list
 * 3. Perform scraping operation for each URL/tag combination
 * 4. Store results (or empty string on failure) in each node
 * 
 * @param configs Vector of ScrapeConfig objects containing URLs and tag boundaries
 * 
 * @note Constructor performs all HTTP operations - execution time depends on network
 * @note Failed scraping operations result in empty string data
 * @note Uses smart pointers for automatic memory management
 */
DataList::DataList(const std::vector<ScrapeConfig>& configs) {
    if (configs.empty()) {
        return; // No configurations to process
    }
    
    // Create the first node and perform initial scraping operation
    head = std::make_unique<Node>();
    head->data = scrape(configs[0]);
    
    // Traverse and create additional nodes for remaining configurations
    Node* current = head.get();
    for (size_t i = 1; i < configs.size(); ++i) {
        current->next = std::make_unique<Node>();  // Create new node with smart pointer
        current = current->next.get();             // Move to the newly created node
        current->data = scrape(configs[i]);        // Perform scraping operation
    }
}

/**
 * @brief Writes all scraped content to the provided output stream
 * 
 * This method traverses the entire linked list and outputs all stored content
 * to the specified stream. It handles empty data gracefully (from failed
 * scraping operations) and maintains the original content formatting.
 * 
 * Output Process:
 * 1. Start from the head of the linked list
 * 2. For each node with valid data, write content to stream
 * 3. Move to the next node
 * 4. Continue until all nodes are processed
 * 
 * @param out Reference to output stream (file, console, stringstream, etc.)
 * 
 * @note Does not add additional formatting - preserves original content structure
 * @note Safely handles nodes with empty data (skips them)
 * @note Can be called multiple times on the same DataList object
 * @note Output stream must be in a valid state for writing
 */
void DataList::write(std::ostream& out) {
    Node* current = head.get();  // Start traversal from the head of the list
    while (current != nullptr) {
        // Only write data if the node contains valid scraped content
        if (!current->data.empty()) {
            out << current->data;  // Write the string content directly
        }
        current = current->next.get();  // Move to the next node in the list
    }
}

/**
 * @brief Extracts content between specified HTML tags from raw HTML text
 * 
 * This method implements the core content extraction logic by searching through
 * HTML content for specified start and end tag pairs. It extracts all text that
 * appears between these boundaries, handling multiple occurrences within the
 * same document.
 * 
 * Extraction Process:
 * 1. Search for start tag using modern string operations
 * 2. When start tag found, collect all subsequent characters
 * 3. Stop collection when end tag is encountered
 * 4. Add separator and continue searching for additional occurrences
 * 5. Return all extracted content as a single string
 * 
 * @param html_content The full HTML source code
 * @param start_tag HTML start tag that marks beginning of desired content
 * @param end_tag HTML end tag that marks end of desired content
 * @return std::string String containing all extracted content
 * 
 * @note Uses value semantics for better memory safety
 * @note Uses std::string operations for better safety and performance
 * @note Handles multiple tag occurrences by continuing search after each match
 * @note Adds OUTPUT_SEPARATOR between different extracted content blocks
 * @note Safe with malformed HTML - won't crash on missing or unmatched tags
 */
std::string DataList::extractContent(const std::string& html_content, 
                                    const std::string& start_tag, 
                                    const std::string& end_tag) {
    std::string result;
    size_t pos = 0;
    
    // Search through the entire HTML content
    while (pos < html_content.length()) {
        // Find the start tag
        size_t start_pos = html_content.find(start_tag, pos);
        if (start_pos == std::string::npos) {
            // No more start tags found
            break;
        }
        
        // Move position to after the start tag
        size_t content_start = start_pos + start_tag.length();
        
        // Find the corresponding end tag
        size_t end_pos = html_content.find(end_tag, content_start);
        if (end_pos == std::string::npos) {
            // No matching end tag found, skip this occurrence
            pos = content_start;
            continue;
        }
        
        // Extract content between tags
        result += html_content.substr(content_start, end_pos - content_start);
        result += OUTPUT_SEPARATOR;
        
        // Move position past the end tag for next search
        pos = end_pos + end_tag.length();
    }

    //clean up any tags that happen to lay between our specified tags

    std::string cleaned_result;

    pos = 0;
    size_t tag_start = 0;

    while(pos < result.length()){

        if (result.find("<", pos) < result.find("#", pos)) {
            tag_start = result.find("<", pos);


            if(tag_start == std::string::npos){
                // No more tags, append rest of string
                cleaned_result += result.substr(pos);
                break;
            }
        
            // Append text before the tag
            cleaned_result += result.substr(pos, tag_start - pos);
        
            // Find end of tag
            size_t tag_end = result.find(">", tag_start);
            if(tag_end == std::string::npos){
                // Malformed HTML, append rest and break
                cleaned_result += result.substr(tag_start);
                break;
            }
        
            // Move past the closing >
            pos = tag_end + 1;
        }else{
            tag_start = result.find("#", pos);


            if(tag_start == std::string::npos){
                // No more tags, append rest of string
                cleaned_result += result.substr(pos);
                break;
            }
        
            // Append text before the tag
            cleaned_result += result.substr(pos, tag_start - pos);
        
            // Find end of tag
            size_t tag_end = result.find(";", tag_start);
            if(tag_end == std::string::npos){
                // Malformed HTML, append rest and break
                cleaned_result += result.substr(tag_start);
                break;
            }

            // Move past the closing 
            pos = tag_end + 1;
        }

    }

    return cleaned_result;
}

/**
 * @brief Modern string comparison function for HTML tag matching
 * 
 * This method searches for a target tag string starting at a given position
 * in HTML content using modern C++ string operations for better safety and
 * performance compared to character-by-character comparison.
 * 
 * Search Process:
 * 1. Use std::string::find() to locate the target tag
 * 2. Check if the tag is found at the current position
 * 3. If found, advance position past the tag
 * 4. Return whether the tag was found at the specified position
 * 
 * @param html_content The HTML content to search in
 * @param pos Starting position to search from (modified to position after match if found)
 * @param target Target tag string to search for
 * @return bool True if tag found at position, false otherwise
 * 
 * @note Modifies 'pos' parameter: advances past match on success
 * @note Uses std::string operations for better safety and performance
 * @note Thread-safe and bounds-safe by design
 * @note More readable and maintainable than character-by-character comparison
 */
bool DataList::findTag(const std::string& html_content, size_t& pos, const std::string& target) {
    // Check if we have enough content remaining for the target tag
    if (pos + target.length() > html_content.length()) {
        return false;
    }
    
    // Check if the target tag matches at the current position
    if (html_content.substr(pos, target.length()) == target) {
        pos += target.length();  // Advance position past the matched tag
        return true;
    }
    
    // No match at current position, advance by one character for next search
    ++pos;
    return false;
}

/**
 * @brief libCurl callback function for receiving HTTP response data
 * 
 * This static method serves as a callback function for libCurl's HTTP operations.
 * libCurl calls this function repeatedly as data is received from the web server,
 * allowing the application to process the incoming data in chunks.
 * 
 * Callback Process:
 * 1. Calculate total number of bytes received (size * nmemb)
 * 2. Cast void pointer to char pointer for string operations
 * 3. Append received data to the output string buffer
 * 4. Return number of bytes processed (signals success to libCurl)
 * 
 * @param contents Pointer to received data buffer from libCurl
 * @param size Size of each data element (typically 1 byte)
 * @param nmemb Number of data elements received
 * @param output Pointer to std::string where data should be accumulated
 * @return size_t Number of bytes processed (must equal size * nmemb for success)
 * 
 * @note Must be static to comply with C callback function signature requirements
 * @note Return value different from (size * nmemb) signals error to libCurl
 * @note Function signature cannot be modified - required by libCurl API
 * @note The output parameter is passed as void* but must be std::string*
 */
size_t DataList::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t total_size = size * nmemb;  // Calculate total bytes received
    output->append(static_cast<char*>(contents), total_size);  // Append data to output string
    return total_size;  // Return bytes processed (signals success to libCurl)
}

/**
 * @brief Core web scraping method that performs HTTP requests and content extraction
 * 
 * This method implements the complete web scraping workflow for a single URL:
 * 1. Display scraping parameters to console for monitoring
 * 2. Initialize libCurl for HTTP operations
 * 3. Configure HTTP request parameters
 * 4. Perform the HTTP GET request
 * 5. Extract content between specified HTML tags
 * 6. Clean up resources and return results
 * 
 * The method provides comprehensive error handling for network failures,
 * HTTP errors, and libCurl initialization problems. Console output uses
 * colored text to clearly show scraping progress and results.
 * 
 * @param config ScrapeConfig containing URL and tag boundaries
 * @return std::string Extracted content, or empty string on failure
 * 
 * @note Uses value semantics for better memory safety
 * @note Performs console output with color coding for status monitoring
 * @note Handles all libCurl cleanup automatically with RAII principles
 * @note Network timeouts and HTTP errors result in empty string return
 */
std::string DataList::scrape(const ScrapeConfig& config) {
    CURL* curl;                    // libCurl handle for HTTP operations
    CURLcode res;                  // Result code from libCurl operations
    std::string html_content;      // Buffer for raw HTML response

    // Display scraping parameters with colored console output for monitoring
    std::cout << '\n' << "\033[35m" << "  url: " << "\033[00m" << config.url << std::endl;
    std::cout << "\033[35m" << "  start: " << "\033[00m" << config.start_tag << std::endl;
    std::cout << "\033[35m" << "  end: " << "\033[00m" << config.end_tag << std::endl;

    // Initialize libCurl global state (required before any libCurl operations)
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl) {
        // Configure the target URL for the HTTP request
        curl_easy_setopt(curl, CURLOPT_URL, config.url.c_str());

        // Configure callback function to receive HTTP response data
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &html_content);

        // Execute the HTTP GET request
        res = curl_easy_perform(curl);

        // Clean up libCurl resources (RAII-style cleanup)
        curl_easy_cleanup(curl);
        curl_global_cleanup();

        // Check for HTTP or network errors
        if (res != CURLE_OK) {
            std::cerr << "'curl_easy_perform' failed: " << curl_easy_strerror(res) << std::endl;
            return "";  // Return empty string on error
        }

        // Extract content between specified HTML tags from the raw response
        std::string parsed_content = extractContent(html_content, config.start_tag, config.end_tag);
        
        std::cout << "\033[32m" << "Success\n" << "\033[0m" << std::endl;
        return parsed_content;
    }

    // Handle cases where libCurl initialization failed
    std::cerr << "Failed to initialize libCurl" << std::endl;
    return "";  // Return empty string on error
}