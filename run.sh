#!/bin/bash

# Color definitions
SUCCESS='\033[38;5;67m'
INFO='\033[38;5;109m'
WARNING='\033[38;5;214m'
ERROR='\033[38;5;167m'
SUBTLE='\033[38;5;245m'
NC='\033[0m'
BOLD='\033[1m'

source .venv/bin/activate

# Parse command line arguments
CONFIG_FILE="config.json"
TRAIN_MODE=false
PREDICT_MODE=false
SENTIMENT_MODE=false
DISCORD_MODE=false
FULL_PIPELINE=true

# Show usage
show_usage() {
    echo -e "${BOLD}GRU Trade Engine${NC}"
    echo ""
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config FILE        Use specific config file (default: config.json)"
    echo "  --train, -t          Train GRU model only"
    echo "  --predict, -p        Run prediction only"
    echo "  --sentiment, -s      Run sentiment analysis only"
    echo "  --discord, -d        Send results to Discord only"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                          # Run full pipeline with config.json"
    echo "  ./run.sh --config custom.json     # Run full pipeline with custom config"
    echo "  ./run.sh --train                  # Train model only"
    echo "  ./run.sh --predict --discord      # Predict and send to Discord"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --train|-t)
            TRAIN_MODE=true
            FULL_PIPELINE=false
            shift
            ;;
        --predict|-p)
            PREDICT_MODE=true
            FULL_PIPELINE=false
            shift
            ;;
        --sentiment|-s)
            SENTIMENT_MODE=true
            FULL_PIPELINE=false
            shift
            ;;
        --discord|-d)
            DISCORD_MODE=true
            FULL_PIPELINE=false
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            # If it's a file, treat it as config file for backward compatibility
            if [ -f "$1" ]; then
                CONFIG_FILE="$1"
                shift
            else
                echo -e "${ERROR}Unknown option: $1${NC}" >&2
                show_usage
                exit 1
            fi
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${ERROR}Config file not found: $CONFIG_FILE${NC}" >&2
    echo ""
    echo "Create it by copying the example:"
    echo "  cp config.example.json $CONFIG_FILE"
    exit 1
fi

# Extract values from config.json using Python
echo -e "${INFO}Loading configuration from: ${BOLD}$CONFIG_FILE${NC}"
stock_ticker=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['stock']['ticker'])")
company_name=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['stock']['company_name'])")
output_dir=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['paths']['output_dir'])")
prediction_out=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['paths']['prediction_output'])")
sentiment_out=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['paths']['sentiment_output'])")
webscraper_exe=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['paths']['webscraper_executable'])")
data_dir=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['paths']['data_dir'])")

echo -e "${SUBTLE}Stock: ${BOLD}$stock_ticker${NC}${SUBTLE} ($company_name)${NC}"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Execute based on mode
if [ "$FULL_PIPELINE" = true ]; then
    echo -e "${SUCCESS}${BOLD}Running Full Pipeline${NC}"
    echo ""

    echo -e "${INFO}[1/4] Running GRU prediction...${NC}"
    python3 main.py --mode p --config "$CONFIG_FILE" > "$prediction_out"

    echo -e "${INFO}[2/4] Running web scraper...${NC}"
    "$(pwd)/$webscraper_exe" "$data_dir" "$company_name"

    echo -e "${INFO}[3/4] Running sentiment analysis...${NC}"
    python3 main.py --mode s --config "$CONFIG_FILE" > "$sentiment_out"

    echo -e "${INFO}[4/4] Sending results to Discord...${NC}"
    python3 main.py --mode discord --config "$CONFIG_FILE"

else
    # Individual modes
    if [ "$TRAIN_MODE" = true ]; then
        echo -e "${INFO}Training GRU model...${NC}"
        python3 main.py --mode t --config "$CONFIG_FILE"
    fi

    if [ "$PREDICT_MODE" = true ]; then
        echo -e "${INFO}Running GRU prediction...${NC}"
        python3 main.py --mode p --config "$CONFIG_FILE" > "$prediction_out"
    fi

    if [ "$SENTIMENT_MODE" = true ]; then
        echo -e "${INFO}Running web scraper...${NC}"
        "$(pwd)/$webscraper_exe" "$data_dir" "$company_name"
        echo -e "${INFO}Running sentiment analysis...${NC}"
        python3 main.py --mode s --config "$CONFIG_FILE" > "$sentiment_out"
    fi

    if [ "$DISCORD_MODE" = true ]; then
        echo -e "${INFO}Sending results to Discord...${NC}"
        python3 main.py --mode discord --config "$CONFIG_FILE"
    fi
fi

echo ""
echo -e "${SUCCESS}${BOLD}✓ Complete!${NC}"


