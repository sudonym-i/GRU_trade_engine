
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