

#include <string>      // For std::string operations
#include <iostream>    // For console input/output operations
#include <fstream>     // For file input/output operations
#include <sstream>     // For string stream operations
#include <vector>      // For std::vector containers
#include <stdexcept>   // For exception handling
#include "scraper.h"   // Custom header for DataList class and scraping functionality
#include <fmt/format.h>  // For fmt::format library




int main(int argc, char* argv[]) {
    // params: webscrape.exe <output_path> <stock_name>

    try {
        // Parse the JSON configuration file to get scraping parameters

        ScrapeConfig config;

        config.start_tag = "<small>";

        config.end_tag = "</small>";

        config.url = fmt::format("https://filmot.com/search/{}%20stock/1?sortField=uploaddate&sortOrder=desc&gridView=1&", std::string(argv[2]));

        // Open output file for writing scraped content
        
        std::string outfile_path = std::string(argv[1]) + "/youtube_data.raw";

        std::ofstream outfile(outfile_path);

        if (!outfile.is_open()) {
            throw std::runtime_error("Failed to open output file: " + outfile_path);
        }

        // Write all scraped and processed content to the output file
        
        DataList data;
        
        outfile << data.scrape(config);

        std::cout << "Successfully wrote scraped data to: " << argv[1] << std::endl;

        outfile.close();

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} // END main