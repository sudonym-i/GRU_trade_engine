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


struct ScrapeConfig {
    std::string url;       ///< Target URL to scrape
    std::string start_tag; ///< HTML start tag for content extraction
    std::string end_tag;   ///< HTML end tag for content extraction
};

const char OUTPUT_SEPARATOR = '\n';


class DataList {
public:

    DataList() = default;

    ~DataList() = default;

    explicit DataList(const std::vector<ScrapeConfig>& configs);

    void write(std::ostream& out);

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output);\

    std::string scrape(const ScrapeConfig& config);

private:

    struct Node {
        std::string data;                    ///< Scraped content
        std::unique_ptr<Node> next = nullptr; ///< Smart pointer to next node
    };

    std::unique_ptr<Node> head = nullptr;

    bool findTag(const std::string& html_content, size_t& pos, const std::string& target);

    std::string extractContent(const std::string& html_content, 
                              const std::string& start_tag, 
                              const std::string& end_tag);
};

#endif // SCRAPER_H