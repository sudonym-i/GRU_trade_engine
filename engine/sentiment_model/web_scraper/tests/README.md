# Web Scraper Test Suite

This directory contains comprehensive unit tests for the ML Trading Bot web scraper components using Google Test framework.

## Test Structure

- **test_main_functions.cpp**: Tests for JSON parsing and configuration functions from main.cpp
- **test_scraper.cpp**: Tests for DataList class and scraping functionality
- **test_runner.cpp**: Main test runner with custom output formatting
- **test_helpers.h**: Helper functions and macros for testing
- **main_functions_for_testing.cpp**: Implementation of testable functions from main.cpp

## Dependencies

- **Google Test**: Automatically downloaded via CMake FetchContent
- **Google Mock**: Included with Google Test for future mocking needs
- **fmt**: For string formatting (already configured in main project)
- **libCurl**: For HTTP functionality testing (already configured)

## Building and Running Tests

### From the web_scraper directory:

```bash
cd /home/sudonym/repos/ML-trading-bot/src/sentiment_model/web_scraper
mkdir build
cd build
cmake ..
make

# Run tests
./tests/scraper_tests

# Run tests with verbose output
./tests/scraper_tests --gtest_output=xml:test_results.xml

# Run specific test suite
./tests/scraper_tests --gtest_filter="JsonParsingTest.*"

# Run tests excluding network tests
SKIP_NETWORK_TESTS=1 ./tests/scraper_tests
```

### Using CMake/CTest:

```bash
# Run all tests through CTest
ctest

# Run tests with verbose output
ctest --verbose

# Run specific tests
ctest -R "scraper_tests"
```

## Test Categories

### Unit Tests
- JSON parsing functions
- HTML content extraction
- Data structure operations
- Input validation and error handling

### Integration Tests (Optional)
- HTTP request functionality
- End-to-end scraping workflows
- Network error handling

**Note**: Integration tests requiring network access are disabled by default. Set `SKIP_NETWORK_TESTS=0` to enable them.

## Test Coverage

The test suite covers:
- ✅ JSON value extraction from configuration files
- ✅ HTML content parsing and extraction
- ✅ Error handling for malformed JSON/HTML
- ✅ Edge cases and boundary conditions
- ✅ Performance testing for large data
- ✅ Memory management and resource cleanup
- 🔄 HTTP callback functionality
- ❌ Network integration tests (optional)

## Adding New Tests

1. Create test functions using Google Test macros:
   ```cpp
   TEST(TestSuite, TestName) {
       // Your test code here
       EXPECT_EQ(expected, actual);
   }
   ```

2. Use test fixtures for complex setup:
   ```cpp
   class MyTestFixture : public ::testing::Test {
   protected:
       void SetUp() override { /* setup code */ }
       void TearDown() override { /* cleanup code */ }
   };
   
   TEST_F(MyTestFixture, TestName) {
       // Test using fixture
   }
   ```

3. Add the test file to CMakeLists.txt if creating a new file

## Troubleshooting

**Common Issues:**

1. **Missing fmt library**: Ensure fmt is installed or available through vcpkg
2. **CURL not found**: Install libcurl development libraries
3. **Network tests failing**: Set `SKIP_NETWORK_TESTS=1` environment variable
4. **Build failures**: Check that all dependencies are properly linked in CMakeLists.txt

**Debug Build:**
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
gdb ./tests/scraper_tests
```

## Best Practices

- Keep tests isolated and independent
- Use descriptive test names that explain what is being tested
- Test both success and failure cases
- Mock external dependencies when possible
- Use test fixtures for common setup/teardown
- Run tests frequently during development