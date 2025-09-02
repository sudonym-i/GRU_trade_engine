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





size_t DataList::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t total_size = size * nmemb;  // Calculate total bytes received
    output->append(static_cast<char*>(contents), total_size);  // Append data to output string
    return total_size;  // Return bytes processed (signals success to libCurl)
}



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