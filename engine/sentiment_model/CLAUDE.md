# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sentiment Model Component

The sentiment_model directory contains a C++ web scraper utility designed to extract content from websites for sentiment analysis data collection. This component is part of the larger ML trading bot project.

## Architecture

### Web Scraper (C++)
- **Purpose**: Extracts content between specified HTML tags from target websites
- **Language**: C++ with libCurl dependency
- **Build System**: CMake
- **Configuration**: Uses `chain.txt` for scraping parameters
- **Output**: Saves scraped data to files in the `/build` directory

### Key Components
- `main.cpp`: Entry point and configuration parser
- `scraper.cpp/.h`: Core scraping functionality using libCurl
- `CMakeLists.txt`: Build configuration with vcpkg toolchain
- `chain.txt`: Configuration file specifying URLs, HTML tags, and output settings

## Development Commands

### Building the Web Scraper
```bash
cd /home/sudonym/repos/ML-trading-bot/src/sentiment_model/web-scraper
mkdir build
cd build
cmake ..
make
```

### Running the Scraper
```bash
cd /home/sudonym/repos/ML-trading-bot/src/sentiment_model/web-scraper/build
./webscrape.exe
```

## Configuration

### Setting Up Scraping Jobs
Edit `chain.txt` with the following format:
```
{
start_tag,end_tag,url
}
OUTPUT_NAME: filename.csv;
```

### Dependencies
- **libCurl**: Required for HTTP requests
- **vcpkg**: Package manager (toolchain path: `/home/isaac/vcpkg/scripts/buildsystems/vcpkg.cmake`)

## Important Notes

- The scraper uses static memory allocation with size limits (MAX_LINE=512, MAX_LENGTH=52)
- JavaScript-heavy sites may require careful tag selection as content might be dynamically loaded
- All scraped output is automatically saved to the `/build` directory
- Configuration syntax is strict - curly brackets must be on separate lines with no spaces

## Troubleshooting

### Build Issues
- Ensure libCurl is installed and accessible
- Verify vcpkg toolchain path in CMakeLists.txt matches your system
- Check that all dependencies are properly linked

### Scraping Issues
- For JavaScript-rendered sites, inspect page source after full load
- Use specific/unique HTML tags as start/end markers
- Verify `chain.txt` syntax follows exact format requirements