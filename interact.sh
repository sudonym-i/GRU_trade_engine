
#!/bin/bash

# Color definitions - Sophisticated complementary palette
PRIMARY='\033[38;5;24m'      # Deep blue
SECONDARY='\033[38;5;67m'    # Muted blue-gray
ACCENT='\033[38;5;143m'      # Soft olive green
SUCCESS='\033[38;5;29m'      # Forest green
WARNING='\033[38;5;214m'     # Warm orange
ERROR='\033[38;5;167m'       # Soft red
INFO='\033[38;5;109m'        # Dusty blue
SUBTLE='\033[38;5;245m'      # Medium gray
BRIGHT='\033[38;5;255m'      # Clean white
DIM_TEXT='\033[38;5;240m'    # Dark gray
HIGHLIGHT='\033[38;5;180m'   # Warm beige
NEUTRAL='\033[38;5;252m'     # Light gray
NC='\033[0m' # No Color

# Legacy color mappings for compatibility
RED=$ERROR
GREEN=$SUCCESS
YELLOW=$WARNING
BLUE=$PRIMARY
PURPLE=$SECONDARY
CYAN=$INFO
WHITE=$BRIGHT
GRAY=$SUBTLE
BLACK='\033[0;30m'
LIGHT_RED=$ERROR
LIGHT_GREEN=$ACCENT
LIGHT_BLUE=$INFO
LIGHT_PURPLE=$SECONDARY
LIGHT_CYAN=$INFO
ORANGE=$WARNING

# Text formatting - Enhanced
BOLD='\033[1m'
UNDERLINE='\033[4m'
DIM='\033[2m'
ITALIC='\033[3m'
BLINK='\033[5m'
REVERSE='\033[7m'
STRIKETHROUGH='\033[9m'

# Background colors for modern effects
BG_BLACK='\033[40m'
BG_RED='\033[41m'
BG_GREEN='\033[42m'
BG_YELLOW='\033[43m'
BG_BLUE='\033[44m'
BG_PURPLE='\033[45m'
BG_CYAN='\033[46m'
BG_WHITE='\033[47m'

# Modern Unicode symbols
ARROW_RIGHT="→"
ARROW_DOWN="↓"
CHECKMARK="✓"
CROSS="✗"
WARNING="⚠"
ROCKET="*"
GEAR="*"
LIGHTNING="*"
SPARKLES="*"
FIRE="*"
BRAIN="*"
CHART="*"
DOLLAR="$"
SHIELD="*"
HOURGLASS="*"
CLOCK="*"
TARGET="*"

# Progress bar function
show_progress() {
    local duration=${1:-3}
    local task_name="$2"
    local width=50
    
    echo -ne "${INFO}${task_name}${NC} ["
    for ((i=0; i<=width; i++)); do
        printf "%*s" $i | tr ' ' "#"
        printf "%*s" $((width-i)) | tr ' ' "_"
        printf "] %d%%" $((i*100/width))
        sleep $(echo "scale=2; $duration/$width" | bc -l)
        printf "\r${INFO}${task_name}${NC} ["
    done
    printf "%*s" $width | tr ' ' "|"
    printf "] 100%%\n"
}

# Animated loading spinner
spinner() {
    local pid=$1
    local task_name="$2"
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf "\r${INFO}[%c${INFO}] ${NEUTRAL}%s...${NC}" "$spinstr" "$task_name"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
    printf "\r${SUCCESS}[${CHECKMARK}] ${NEUTRAL}%s complete!${NC}\n" "$task_name"
}

# Terminal wave effect function
wave_effect() {
    local text="$1"
    local colors=("${PRIMARY}" "${SECONDARY}" "${ACCENT}" "${SUCCESS}" "${INFO}" "${WARNING}" "${HIGHLIGHT}")
    
    for i in {1..3}; do
        for color in "${colors[@]}"; do
            echo -ne "\r${color}${BOLD}${text}${NC}"
            sleep 0.1
        done
    done
    echo -e "\r${INFO}${BOLD}${text}${NC}"
}

# Gradient text effect
gradient_text() {
    local text="$1"
    local len=${#text}
    local colors=("${ERROR}" "${WARNING}" "${HIGHLIGHT}" "${ACCENT}" "${INFO}" "${PRIMARY}" "${SECONDARY}")
    local color_count=${#colors[@]}
    
    for ((i=0; i<len; i++)); do
        local color_index=$((i * color_count / len))
        echo -ne "${colors[$color_index]}${text:$i:1}"
    done
    echo -e "${NC}"
}

# Clear screen for better presentation
clear

# ASCII Art Header
echo -e "\n"
echo -e "${PRIMARY}${BOLD}    ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗         ${NC}"
echo -e "${PRIMARY}${BOLD}    ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║         ${NC}"
echo -e "${PRIMARY}${BOLD}    ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║         ${NC}"
echo -e "${PRIMARY}${BOLD}    ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║         ${NC}"
echo -e "${PRIMARY}${BOLD}    ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗    ${NC}"
echo -e "${PRIMARY}${BOLD}    ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ${NC}"
echo -e "\n"
echo -e "${SECONDARY}${BOLD}████████╗██████╗  █████╗ ██████╗ ███████╗    ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗${NC}"
echo -e "${SECONDARY}${BOLD}╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝${NC}"
echo -e "${SECONDARY}${BOLD}   ██║   ██████╔╝███████║██║  ██║█████╗      █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  ${NC}"
echo -e "${SECONDARY}${BOLD}   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝      ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  ${NC}"
echo -e "${SECONDARY}${BOLD}   ██║   ██║  ██║██║  ██║██████╔╝███████╗    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗${NC}"
echo -e "${SECONDARY}${BOLD}   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝    ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝${NC}"
echo -e "\n"
echo -e "${DIM_TEXT}                                      AI-Powered Trading System                                     ${NC}"
echo -e "\n"

# Animated initialization message
echo -ne "${HOURGLASS} "
echo "Initializing automated trading system"
echo -ne "${GRAY}${DIM}"
for i in {1..3}; do
    echo -ne "."
    sleep 0.5
done
echo -e "${NC}\n"

# System status display
echo -e "${SUCCESS}${BOLD}╭─ SYSTEM STATUS ──────────────────────────────╮${NC}"
echo -e "${SUCCESS}${BOLD}│${NC} ${CHECKMARK} ${SUBTLE}Trading Engine:${NC} ${SUCCESS}Ready${NC}                      ${SUCCESS}${BOLD}│${NC}"
echo -e "${SUCCESS}${BOLD}│${NC} ${CHECKMARK} ${SUBTLE}Neural Networks:${NC} ${SUCCESS}Standby${NC}                   ${SUCCESS}${BOLD}│${NC}"
echo -e "${SUCCESS}${BOLD}│${NC} ${CHECKMARK} ${SUBTLE}Market Interface:${NC} ${SUCCESS}Online${NC}                   ${SUCCESS}${BOLD}│${NC}"
echo -e "${SUCCESS}${BOLD}╰──────────────────────────────────────────────╯${NC}"
echo -e "\n"
echo -e "\n"
# Save current directory, hopefully help with the path issues
here=$(pwd)

# May need to automate model downloading as well

echo -e "${LIGHTNING}${YELLOW}${BOLD} STEP 1/4 ${NC}${INFO}${BOLD}Setting up Python virtual environment${NC}"
echo -e "${INFO}╭──────────────────────────────────────────────────────╮${NC}"
echo -e "${INFO}│${NC} ${GEAR} Preparing Python environment...                    ${INFO}│${NC}"
echo -e "${INFO}╰──────────────────────────────────────────────────────╯${NC}"
sudo apt install python3-full > /dev/null 2>&1 &
echo -ne "${CYAN}${BOLD}Handling Python3-full...${NC}"

PID=$!
spinner $PID "Handling Python3-full"

show_progress 2 "${GEAR} Creating virtual environment"
if [ ! -d "$here/.venv" ]; then
    python3 -m venv "$here/.venv" > /dev/null 2>&1
fi

show_progress 1 "${LIGHTNING} Activating virtual environment"
source "$here/.venv/bin/activate"
echo -e "\n"
echo -e "\n"
echo -e "\n${SUCCESS}${BOLD}╭─ STEP 1 COMPLETE ───────────────────────────────────╮${NC}"
echo -e "${SUCCESS}${BOLD}│${NC} ${CHECKMARK} Virtual environment ready                         ${SUCCESS}${BOLD}│${NC}"
echo -e "${SUCCESS}${BOLD}│${NC} ${CHECKMARK} Python3 installed and configured                  ${SUCCESS}${BOLD}│${NC}"
echo -e "${SUCCESS}${BOLD}╰─────────────────────────────────────────────────────╯${NC}"
echo ""
echo -e "${LIGHTNING}${YELLOW}${BOLD} STEP 2/4 ${NC}${LIGHT_BLUE}${BOLD}Handling Python dependencies${NC}"
echo -e "${LIGHT_CYAN}╭──────────────────────────────────────────────────────╮${NC}"
echo -e "${LIGHT_CYAN}│${NC} ${GEAR} Handling required Python packages...               ${LIGHT_CYAN}│${NC}"
echo -e "${LIGHT_CYAN}╰──────────────────────────────────────────────────────╯${NC}"

echo -ne "${PURPLE}${BOLD}Reading requirements.txt...${NC}"
echo ""
show_progress 1 "${GEAR} Analyzing dependencies"

echo -ne "${CYAN}${BOLD}Handling packages...${NC}"
pip install -r "$here/backend_&_algorithms/engine/requirements.txt" > /dev/null 2>&1 &
PID=$!
spinner $PID "Handling Python dependencies"
echo -e "\n"
echo -e "\n"
echo -e "\n${GREEN}${BOLD}╭─ STEP 2 COMPLETE ───────────────────────────────────╮${NC}"
echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Python dependencies installed                     ${GREEN}${BOLD}│${NC}"
echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} All packages verified and ready                   ${GREEN}${BOLD}│${NC}"
echo -e "${GREEN}${BOLD}╰─────────────────────────────────────────────────────╯${NC}"
echo ""
echo -e "${LIGHTNING}${YELLOW}${BOLD} STEP 3/4 ${NC}${LIGHT_BLUE}${BOLD}Setting up C++ environment${NC}"
echo -e "${LIGHT_CYAN}╭──────────────────────────────────────────────────────╮${NC}"
echo -e "${LIGHT_CYAN}│${NC} ${GEAR} Configuring C++ build environment...               ${LIGHT_CYAN}│${NC}"
echo -e "${LIGHT_CYAN}╰──────────────────────────────────────────────────────╯${NC}"

show_progress 1 "${GEAR} Creating build directory"
mkdir -p "$here/backend_&_algorithms/engine/sentiment_model/web_scraper/build"

show_progress 3 "${GEAR} Updating package repositories"
sudo apt-get update > /dev/null 2>&1 && sudo apt upgrade -y > /dev/null 2>&1

echo -ne "${PURPLE}${BOLD}Handling C++ dependencies...${NC}\n"

show_progress 2 "${GEAR} Handling CMake"
sudo apt install cmake -y > /dev/null 2>&1

show_progress 2 "${GEAR} Handling Google Test"
sudo apt-get install libgtest-dev -y > /dev/null 2>&1

show_progress 2 "${GEAR} Handling fmt library"
sudo apt install libfmt-dev -y > /dev/null 2>&1

show_progress 2 "${GEAR} Handling cURL"
sudo apt install curl -y > /dev/null 2>&1

show_progress 2 "${GEAR} Handling cURL development libraries"
sudo apt install libcurl4-gnutls-dev -y > /dev/null 2>&1

echo -e "\n${FIRE}${PURPLE}${BOLD}Compiling C++ components...${NC}"
cd "$here/backend_&_algorithms/engine/sentiment_model/web_scraper/build"

show_progress 3 "${GEAR} Running CMake configuration"
cmake .. > /dev/null 2>&1

show_progress 4 "${FIRE} Building C++ project"
make > /dev/null 2>&1

show_progress 1 "${CHECKMARK} Setting executable permissions"
chmod +x webscrape.exe

cd "$here"
echo -e "\n"
echo -e "\n"
echo -e "\n${GREEN}${BOLD}╭─ STEP 3 COMPLETE ───────────────────────────────────╮${NC}"
echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} C++ environment configured                        ${GREEN}${BOLD}│${NC}"
echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Web scraper compiled and ready                    ${GREEN}${BOLD}│${NC}"
echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} All C++ dependencies installed                    ${GREEN}${BOLD}│${NC}"
echo -e "${GREEN}${BOLD}╰─────────────────────────────────────────────────────╯${NC}"

echo -e "${SPARKLES}${YELLOW}${BOLD} STEP 4/4 ${NC}${LIGHT_GREEN}${BOLD}Setup complete!${NC} ${CHECKMARK}"
echo -e "\n"
echo -e "\n"
echo -e "\n${LIGHT_PURPLE}${BOLD}╭══════════════════════════════════════════════════╮${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}║${NC}                                                  ${LIGHT_PURPLE}${BOLD}║${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}║${NC}      ${GEAR} ${LIGHT_CYAN}${BOLD}Configuration Phase${NC} ${TARGET}                     ${LIGHT_PURPLE}${BOLD}║${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}║${NC}    ${GRAY}${DIM}Customize your trading parameters${NC}             ${LIGHT_PURPLE}${BOLD}║${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}║${NC}                                                  ${LIGHT_PURPLE}${BOLD}║${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}╰══════════════════════════════════════════════════╯${NC}"
echo -e "\n"
echo -e "\n"
echo -e "${TARGET}${YELLOW}${BOLD} CONFIGURATION ${NC}"
echo -e "${LIGHT_BLUE}${BOLD}╭─────────────── Trading Parameters ──────────────╮${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC}                                                 ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC} ${CHART} ${YELLOW}What ticker symbol to trade?${NC}                  ${LIGHT_BLUE}${BOLD}│${NC}"
read -p "     Enter ticker (e.g., AAPL, TSLA): " ticker

echo -e "${LIGHT_BLUE}${BOLD}│${NC}                                                 ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC} ${BRAIN} ${YELLOW}Semantic name for AI analysis:${NC}                ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC}   ${GRAY}${DIM}(e.g., GOOG → google, AAPL → apple)${NC}           ${LIGHT_BLUE}${BOLD}│${NC}"
read -p "     Semantic name: " semantic_name

echo -e "${LIGHT_BLUE}${BOLD}│${NC}                                                 ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC} ${CLOCK} ${WHITE}Trading time interval:${NC}                        ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC}   ${LIGHT_GREEN}1${NC} ${ARROW_RIGHT} 1hr              ${LIGHT_GREEN}2${NC} ${ARROW_RIGHT} 5min   ${LIGHT_GREEN}3${NC} ${ARROW_RIGHT} 15min     ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC}   ${LIGHT_GREEN}4${NC} ${ARROW_RIGHT} 30min            ${LIGHT_GREEN}5${NC} ${ARROW_RIGHT} 1day                 ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC}                                                 ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}╰─────────────────────────────────────────────────╯${NC}"
read -p "     Enter choice (1-5): " time_interval
echo -e "\n"
echo -e "\n"
# Convert numeric choice to actual time interval
case $time_interval in
    1) interval_value="1hr" ;;
    2) interval_value="5min" ;;
    3) interval_value="15min" ;;
    4) interval_value="30min" ;;
    5) interval_value="1d" ;;
    *) interval_value="1hr" ;; # default
esac

echo -e "\n${BRAIN}${LIGHT_PURPLE}${BOLD} AI MODEL TRAINING ${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}╭─────────────── Model Configuration ──────────────╮${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}│${NC}                                                  ${LIGHT_PURPLE}${BOLD}│${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}│${NC} ${SPARKLES} ${WHITE}Train sentiment analysis model?${NC}                ${LIGHT_PURPLE}${BOLD}│${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}│${NC}   ${GRAY}${DIM}(Recommended for better accuracy)${NC}              ${LIGHT_PURPLE}${BOLD}│${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}│${NC}                                                  ${LIGHT_PURPLE}${BOLD}│${NC}"
echo -e "${LIGHT_PURPLE}${BOLD}╰──────────────────────────────────────────────────╯${NC}"
read -p "     Train sentiment model? (y/n): " train_models
echo -e "\n"
echo -e "\n"


# if these directories don't exist, create them
if [ ! -d "$here/backend_&_algorithms/engine/sentiment_model/raw_data" ]; then
    mkdir "$here/backend_&_algorithms/engine/sentiment_model/raw_data"
fi

if [ ! -d "$here/backend_&_algorithms/engine/sentiment_model/processed_data" ]; then
    mkdir "$here/backend_&_algorithms/engine/sentiment_model/processed_data"
fi

echo -e "\n${CHART}${LIGHT_BLUE}${BOLD} DATA COLLECTION ${NC}"
echo -e "${LIGHT_BLUE}${BOLD}╭──────────────── Gathering Market Data ─────────────────╮${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC}                                                        ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC} ${LIGHTNING} Scraping web data...                                 ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}│${NC}                                                        ${LIGHT_BLUE}${BOLD}│${NC}"
echo -e "${LIGHT_BLUE}${BOLD}╰────────────────────────────────────────────────────────╯${NC}"

cd "$here/backend_&_algorithms/"

show_progress 3 "${GEAR} Initializing web scraper"
show_progress 5 "${CHART} Collecting market sentiment data"

"$here/.venv/bin/python3" "main.py" webscrape --ticker "$semantic_name" > /dev/null 2>&1 &
PID=$!
spinner $PID "Scraping financial news and sentiment data"

echo -e "\n${GREEN}${BOLD}╭─ DATA COLLECTION COMPLETE ────────────────────────────────╮${NC}"
echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Web scraping successful                                 ${GREEN}${BOLD}│${NC}"
echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Market data collected and processed                     ${GREEN}${BOLD}│${NC}"
echo -e "${GREEN}${BOLD}╰───────────────────────────────────────────────────────────╯${NC}"
echo -e "\n"
echo -e "\n"
if [ "$train_models" == "y" ] || [ "$train_models" == "Y" ]; then
    echo -e "\n${BRAIN}${LIGHT_PURPLE}${BOLD} SENTIMENT MODEL TRAINING ${NC}"
    echo -e "${LIGHT_PURPLE}${BOLD}╭─────────────── Training Neural Network ──────────────╮${NC}"
    echo -e "${LIGHT_PURPLE}${BOLD}│${NC}                                                      ${LIGHT_PURPLE}${BOLD}│${NC}"
    echo -e "${LIGHT_PURPLE}${BOLD}│${NC} ${FIRE} Training deep learning sentiment model...          ${LIGHT_PURPLE}${BOLD}│${NC}"
    echo -e "${LIGHT_PURPLE}${BOLD}│${NC}     ${GRAY}${DIM}This process will take several minutes${NC}           ${LIGHT_PURPLE}${BOLD}│${NC}"
    echo -e "${LIGHT_PURPLE}${BOLD}│${NC}                                                      ${LIGHT_PURPLE}${BOLD}│${NC}"
    echo -e "${LIGHT_PURPLE}${BOLD}╰──────────────────────────────────────────────────────╯${NC}"
    
    cd "$here/backend_&_algorithms/engine/sentiment_model/"
    
    show_progress 4 "${CHART} Downloading training dataset"
    "$here/.venv/bin/python3" "$here/backend_&_algorithms/engine/sentiment_model/download_dataset.py" > /dev/null 2>&1
    
    show_progress 6 "${GEAR} Tokenizing and preprocessing data"
    "$here/.venv/bin/python3" "$here/backend_&_algorithms/engine/sentiment_model/tokenize_pipeline.py" > /dev/null 2>&1 &
    PID=$!
    spinner $PID "Processing text data and creating tokens"
    
    show_progress 8 "${BRAIN} Training neural network"
    echo -e "\n${HOURGLASS}${YELLOW}${BOLD} This may take 10-15 minutes depending on your hardware...${NC}\n"
    
    "$here/.venv/bin/python3" "$here/backend_&_algorithms/engine/sentiment_model/train.py" &
    PID=$!
    spinner $PID "Training deep learning model"
    
    echo -e "\n${GREEN}${BOLD}╭─ SENTIMENT MODEL TRAINING COMPLETE ───────────────────────╮${NC}"
    echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Neural network trained successfully                     ${GREEN}${BOLD}│${NC}"
    echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Model ready for sentiment analysis                      ${GREEN}${BOLD}│${NC}"
    echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} AI accuracy optimized for trading decisions             ${GREEN}${BOLD}│${NC}"
    echo -e "${GREEN}${BOLD}╰───────────────────────────────────────────────────────────╯${NC}"
fi

cd "$here"
echo -e "\n"
echo -e "\n"

echo -e "\n${TARGET}${LIGHT_RED}${BOLD} TSR MODEL TRAINING ${NC}"
echo -e "${LIGHT_RED}${BOLD}╭─────────────── Time Series Regression Model ─────────────╮${NC}"
echo -e "${LIGHT_RED}${BOLD}│${NC}                                                          ${LIGHT_RED}${BOLD}│${NC}"
echo -e "${LIGHT_RED}${BOLD}│${NC} ${ROCKET} ${WHITE}Train GRU prediction model?${NC}                            ${LIGHT_RED}${BOLD}│${NC}"
echo -e "${LIGHT_RED}${BOLD}│${NC}     ${GRAY}${DIM}(Required for autonomous trading)${NC}                    ${LIGHT_RED}${BOLD}│${NC}"
echo -e "${LIGHT_RED}${BOLD}│${NC}                                                          ${LIGHT_RED}${BOLD}│${NC}"
echo -e "${LIGHT_RED}${BOLD}╰──────────────────────────────────────────────────────────╯${NC}"
read -p "     Train TSR model? (y/n): " train_tsr
echo -e "\n"
echo -e "\n"

cd "$here/backend_&_algorithms/"
if [ "$train_tsr" == "y" ] || [ "$train_tsr" == "Y" ]; then
    echo -e "\n${CHART}${LIGHT_RED}${BOLD} GRU MODEL TRAINING ${NC}"
    echo -e "${LIGHT_RED}${BOLD}╭──────────── Time Series Regression Training ─────────────╮${NC}"
    echo -e "${LIGHT_RED}${BOLD}│${NC}                                                          ${LIGHT_RED}${BOLD}│${NC}"
    echo -e "${LIGHT_RED}${BOLD}│${NC} ${FIRE} Training GRU model for ${NC}                                ${LIGHT_RED}${BOLD}│${NC}"
    echo -e "${LIGHT_RED}${BOLD}│${NC}     ${GRAY}${DIM}Analyzing price patterns and trends${NC}                  ${LIGHT_RED}${BOLD}│${NC}"
    echo -e "${LIGHT_RED}${BOLD}│${NC}                                                          ${LIGHT_RED}${BOLD}│${NC}"
    echo -e "${LIGHT_RED}${BOLD}╰──────────────────────────────────────────────────────────╯${NC}"
    
    show_progress 3 "${CHART} Fetching historical price data"
    show_progress 5 "${GEAR} Preprocessing time series data"
    show_progress 7 "${BRAIN} Training regression model"
    
    "$here/.venv/bin/python3" main.py train --ticker "$ticker" &
    PID=$!

    echo -e "\n${HOURGLASS}${YELLOW}${BOLD} This may take 10-30 minutes depending on your hardware...${NC}\n"


    spinner $PID "Training GRU prediction model"
    
    echo -e "\n${GREEN}${BOLD}╭─ GRU MODEL TRAINING COMPLETE ──────────────────────────────╮${NC}"
    echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} GRU model trained successfully                           ${GREEN}${BOLD}│${NC}"
    echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Price prediction algorithm optimized                     ${GREEN}${BOLD}│${NC}"
    echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Ready for autonomous trading decisions                   ${GREEN}${BOLD}│${NC}"
    echo -e "${GREEN}${BOLD}╰────────────────────────────────────────────────────────────╯${NC}"
fi
echo -e "\n"
echo -e "\n"
cd $here/integrations_\&_strategy
# Virtual environment activated via direct python path

echo -e "\n${DOLLAR}${LIGHT_GREEN}${BOLD} TRADING SETUP ${NC}"
echo -e "${LIGHT_GREEN}${BOLD}╭══════════════════════════════════════════════════════════╮${NC}"
echo -e "${LIGHT_GREEN}${BOLD}║${NC}                                                          ${LIGHT_GREEN}${BOLD}║${NC}"
echo -e "${LIGHT_GREEN}${BOLD}║${NC}      ${ROCKET} ${LIGHT_CYAN}${BOLD}Ready to Launch Trading Engine${NC} ${TARGET}                  ${LIGHT_GREEN}${BOLD}║${NC}"
echo -e "${LIGHT_GREEN}${BOLD}║${NC}         ${GRAY}${DIM}All systems configured and operational${NC}           ${LIGHT_GREEN}${BOLD}║${NC}"
echo -e "${LIGHT_GREEN}${BOLD}║${NC}                                                          ${LIGHT_GREEN}${BOLD}║${NC}"
echo -e "${LIGHT_GREEN}${BOLD}╰══════════════════════════════════════════════════════════╯${NC}"
echo ""

echo -e "${LIGHT_GREEN}${BOLD}╭─────────────── Launch Configuration ──────────────────╮${NC}"
echo -e "${LIGHT_GREEN}${BOLD}│${NC}                                                       ${LIGHT_GREEN}${BOLD}│${NC}"
echo -e "${LIGHT_GREEN}${BOLD}│${NC} ${ROCKET} ${WHITE}Start automated trading now?${NC}                        ${LIGHT_GREEN}${BOLD}│${NC}"
echo -e "${LIGHT_GREEN}${BOLD}│${NC}                                                       ${LIGHT_GREEN}${BOLD}│${NC}"
echo -e "${LIGHT_GREEN}${BOLD}╰───────────────────────────────────────────────────────╯${NC}"
read -p "     Start trading? (y/n): " start
if [ "$start" == "y" ] || [ "$start" == "Y" ]; then
    echo ""
    echo -e "${DOLLAR}${LIGHT_CYAN}${BOLD} TRADING MODE SELECTION ${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}╭──────────────── Choose Trading Mode ───────────────────╮${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC}                                                        ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC} ${SHIELD} ${LIGHT_GREEN}1${NC} ${ARROW_RIGHT} ${LIGHT_GREEN}Simulation Mode${NC}                                  ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC}     ${GRAY}${DIM}Paper trading - safe testing environment${NC}           ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC}                                                        ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC} ${GEAR} ${ORANGE}2${NC} ${ARROW_RIGHT} ${ORANGE}IB Paper Trading${NC}                                 ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC}     ${GRAY}${DIM}Interactive Brokers paper account${NC}                  ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC}                                                        ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC} ${WARNING} ${LIGHT_RED}3${NC} ${ARROW_RIGHT} ${LIGHT_RED}IB Live Trading${NC}                                  ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC}     ${RED}${DIM}Real money - use with extreme caution!${NC}             ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}│${NC}                                                        ${LIGHT_CYAN}${BOLD}│${NC}"
    echo -e "${LIGHT_CYAN}${BOLD}╰────────────────────────────────────────────────────────╯${NC}"
    echo ""
    read -p "     Enter your choice (1-3): " mode_choice
    echo -e "\n"
    echo -e "\n"
    case $mode_choice in
        1)
            echo -e "\n${SHIELD}${LIGHT_GREEN}${BOLD} LAUNCHING SIMULATION MODE ${NC}"
            echo -e "${LIGHT_GREEN}${BOLD}╭─────────────── Safe Trading Environment ───────────────╮${NC}"
            echo -e "${LIGHT_GREEN}${BOLD}│${NC}                                                        ${LIGHT_GREEN}${BOLD}│${NC}"
            echo -e "${LIGHT_GREEN}${BOLD}│${NC} ${DOLLAR} Virtual portfolio: ${YELLOW}\$10,000${NC} starting capital          ${LIGHT_GREEN}${BOLD}│${NC}"
            echo -e "${LIGHT_GREEN}${BOLD}│${NC} ${CHECKMARK} No real money at risk                                ${LIGHT_GREEN}${BOLD}│${NC}"
            echo -e "${LIGHT_GREEN}${BOLD}│${NC} ${CHART} Perfect for testing strategies                       ${LIGHT_GREEN}${BOLD}│${NC}"
            echo -e "${LIGHT_GREEN}${BOLD}│${NC}                                                        ${LIGHT_GREEN}${BOLD}│${NC}"
            echo -e "${LIGHT_GREEN}${BOLD}╰────────────────────────────────────────────────────────╯${NC}"
            
            show_progress 2 "${GEAR} Configuring simulation parameters"
            # Update config.json with ticker, semantic_name, mode and time_interval
            jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "simulation" --arg interval "$interval_value" \
                '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                config.json > config.json.tmp && mv config.json.tmp config.json
            echo -e "${CHECKMARK} Opening configuration editor..."
            nano config.json
            
            show_progress 3 "${ROCKET} Launching trading engine"
            echo -e "\n${SPARKLES}${LIGHT_GREEN}${BOLD} SIMULATION MODE ACTIVE! ${NC}"
            $here/.venv/bin/python3 schedule_trader.py --start
            ;;
        2)
            echo -e "\n${GEAR}${ORANGE}${BOLD} LAUNCHING IB PAPER TRADING ${NC}"
            echo -e "${ORANGE}${BOLD}╭─────────── Interactive Brokers Connection ─────────────╮${NC}"
            echo -e "${ORANGE}${BOLD}│${NC}                                                       ${ORANGE}${BOLD}│${NC}"
            echo -e "${ORANGE}${BOLD}│${NC} ${LIGHTNING} Testing IB Gateway connection...                    ${ORANGE}${BOLD}│${NC}"
            echo -e "${ORANGE}${BOLD}│${NC} ${SHIELD} Paper trading account - no real money risk          ${ORANGE}${BOLD}│${NC}"
            echo -e "${ORANGE}${BOLD}│${NC}                                                       ${ORANGE}${BOLD}│${NC}"
            echo -e "${ORANGE}${BOLD}╰────────────────────────────────────────────────────────╯${NC}"
            
            show_progress 2 "${GEAR} Configuring IB paper parameters"
            echo -e "\n"
            echo -e "\n"
            # Update config.json with ticker, semantic_name, mode and time_interval
            jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "ib_paper" --arg interval "$interval_value" \
                '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                config.json > config.json.tmp && mv config.json.tmp config.json
            show_progress 3 "${LIGHTNING} Testing IB connection"
            $here/.venv/bin/python3 test_ib_connection.py --mode paper > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo -e "\n${GREEN}${BOLD}╭─ IB CONNECTION SUCCESSFUL ─────────────────────────────────╮${NC}"
                echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Interactive Brokers connection established          ${GREEN}${BOLD}│${NC}"
                echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Paper trading mode activated                        ${GREEN}${BOLD}│${NC}"
                echo -e "${GREEN}${BOLD}╰────────────────────────────────────────────────────────────╯${NC}"
                
                show_progress 2 "${ROCKET} Launching IB paper trading"
                echo -e "\n${SPARKLES}${ORANGE}${BOLD} IB PAPER TRADING ACTIVE! ${NC}"
                $here/.venv/bin/python3 schedule_trader.py --start
            else
                echo -e "\n${CROSS}${RED}${BOLD} IB CONNECTION FAILED ${NC}"
                echo -e "${RED}${BOLD}╭─────────────── Connection Troubleshooting ──────────────╮${NC}"
                echo -e "${RED}${BOLD}│${NC}                                                        ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}│${NC} ${WARNING} ${YELLOW}Please verify the following:${NC}                          ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}│${NC}   ${GRAY}• IB Gateway or TWS is running${NC}                        ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}│${NC}   ${GRAY}• API is enabled in IB settings${NC}                       ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}│${NC}   ${GRAY}• Paper trading port 7496 is configured${NC}               ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}│${NC}                                                        ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}╰─────────────────────────────────────────────────────────╯${NC}"
                
                show_progress 2 "${GEAR} Switching to simulation mode"
                echo -e "\n${SHIELD}${YELLOW}${BOLD} FALLBACK: SIMULATION MODE ACTIVE ${NC}"
                # Update config back to simulation mode
                jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "simulation" --arg interval "$interval_value" \
                    '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                    config.json > config.json.tmp && mv config.json.tmp config.json
                nano config.json
                $here/.venv/bin/python3 schedule_trader.py --start
            fi
            ;;
        3)
            echo -e "\n${WARNING}${RED}${BOLD} LIVE TRADING MODE WARNING ${NC}"
            echo -e "${RED}${BOLD}╭═══════════════════════════════════════════════════════════╮${NC}"
            echo -e "${RED}${BOLD}║${NC}                    ${BLINK}${RED}${BOLD}DANGER ZONE${NC}                    ${RED}${BOLD}║${NC}"
            echo -e "${RED}${BOLD}╠═══════════════════════════════════════════════════════════╣${NC}"
            echo -e "${RED}${BOLD}║${NC}                                                       ${RED}${BOLD}║${NC}"
            echo -e "${RED}${BOLD}║${NC} ${DOLLAR} ${YELLOW}This will execute trades with REAL MONEY!${NC}       ${RED}${BOLD}║${NC}"
            echo -e "${RED}${BOLD}║${NC} ${WARNING} ${YELLOW}All trades occur in your IB live account${NC}        ${RED}${BOLD}║${NC}"
            echo -e "${RED}${BOLD}║${NC} ${CROSS} ${YELLOW}Losses can occur - trade responsibly${NC}            ${RED}${BOLD}║${NC}"
            echo -e "${RED}${BOLD}║${NC}                                                       ${RED}${BOLD}║${NC}"
            echo -e "${RED}${BOLD}╰═══════════════════════════════════════════════════════════╯${NC}"
            echo ""
            echo -e "${RED}${BOLD}╭──────────── Final Confirmation Required ──────────────╮${NC}"
            echo -e "${RED}${BOLD}│${NC}                                                     ${RED}${BOLD}│${NC}"
            echo -e "${RED}${BOLD}│${NC} Type 'yes' to proceed with live trading:           ${RED}${BOLD}│${NC}"
            echo -e "${RED}${BOLD}│${NC}                                                     ${RED}${BOLD}│${NC}"
            echo -e "${RED}${BOLD}╰─────────────────────────────────────────────────────────╯${NC}"
            read -p "     Confirmation: " confirm
            if [ "$confirm" == "yes" ]; then
                echo -e "\n${LIGHTNING}${RED}${BOLD} INITIALIZING LIVE TRADING MODE ${NC}"
                echo -e "${RED}${BOLD}╭────────────── Live Connection Setup ───────────────╮${NC}"
                echo -e "${RED}${BOLD}│${NC}                                                  ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}│${NC} ${GEAR} Testing IB live connection...               ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}│${NC} ${DOLLAR} Preparing real money trading interface     ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}│${NC}                                                  ${RED}${BOLD}│${NC}"
                echo -e "${RED}${BOLD}╰──────────────────────────────────────────────────────╯${NC}"
                
                show_progress 2 "${GEAR} Configuring live trading parameters"
                # Update config.json with ticker, semantic_name, mode and time_interval
                jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "ib_live" --arg interval "$interval_value" \
                    '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                    config.json > config.json.tmp && mv config.json.tmp config.json
                
                echo -e "${CHECKMARK} Opening configuration editor..."
                nano config.json
                
                show_progress 3 "${LIGHTNING} Testing live IB connection"
                $here/.venv/bin/python3 test_ib_connection.py --mode live > /dev/null 2>&1
                if [ $? -eq 0 ]; then
                    echo -e "\n${GREEN}${BOLD}╭─ LIVE CONNECTION SUCCESSFUL ───────────────────────────────╮${NC}"
                    echo -e "${GREEN}${BOLD}│${NC} ${CHECKMARK} Interactive Brokers live connection established     ${GREEN}${BOLD}│${NC}"
                    echo -e "${GREEN}${BOLD}│${NC} ${DOLLAR} Real money trading mode activated                   ${GREEN}${BOLD}│${NC}"
                    echo -e "${GREEN}${BOLD}│${NC} ${WARNING} Trade responsibly - losses are possible             ${GREEN}${BOLD}│${NC}"
                    echo -e "${GREEN}${BOLD}╰────────────────────────────────────────────────────────────╯${NC}"
                    
                    show_progress 2 "${ROCKET} Launching live trading engine"
                    echo -e "\n${ROCKET}${RED}${BOLD} LIVE TRADING ACTIVE - GOOD LUCK! ${NC}"
                    $here/.venv/bin/python3 schedule_trader.py --start
                else
                    echo -e "\n${RED}${BOLD}✗ IB live connection test failed${NC}"
                    echo -e "  ${YELLOW}Please check:${NC}"
                    echo -e "    ${GRAY}• IB Gateway or TWS is running${NC}"
                    echo -e "    ${GRAY}• API is enabled in IB settings${NC}"
                    echo -e "    ${GRAY}• Live trading port 7497 is configured${NC}"
                    echo -e "    ${GRAY}• You have live trading permissions${NC}"
                    echo -e "\n${RED}${BOLD}🛑 Aborting live trading for safety${NC}"
                fi
            else
                echo -e "\n${YELLOW}${BOLD}↻ Live trading cancelled. Falling back to simulation mode...${NC}"
                # Update config.json with simulation mode
                jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "simulation" --arg interval "$interval_value" \
                    '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                    config.json > config.json.tmp && mv config.json.tmp config.json
                
                $here/.venv/bin/python3 automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            fi
            ;;
        *)
            echo -e "\n${YELLOW}${BOLD}⚠ Invalid choice. Defaulting to simulation mode...${NC}"
            # Update config.json with ticker, semantic_name, simulation mode and time_interval
            jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "simulation" --arg interval "$interval_value" \
                '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                config.json > config.json.tmp && mv config.json.tmp config.json
            
            $here/.venv/bin/python3 automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            ;;
    esac
else
    echo -e "\n${GREEN}${BOLD}✓ Setup complete!${NC}"
    echo -e "\n${WHITE}${BOLD}Available trading commands:${NC}"
    echo -e "  ${GREEN}Simulation:${NC}     ${GRAY}$here/.venv/bin/python3 automated_trader.py --mode simulation --stock $ticker --semantic-name $semantic_name${NC}"
    echo -e "  ${YELLOW}IB Paper:${NC}       ${GRAY}$here/.venv/bin/python3 automated_trader.py --mode ib_paper --stock $ticker --semantic-name $semantic_name${NC}"
    echo -e "  ${RED}IB Live:${NC}        ${GRAY}$here/.venv/bin/python3 automated_trader.py --mode ib_live --stock $ticker --semantic-name $semantic_name${NC}"
    echo ""
fi
