
#!/bin/bash

# Color definitions 
ACCENT='\033[38;5;24m'      # Deep blue
SUCCESS='\033[38;5;67m'    # Muted blue-gray
PRIMARY='\033[38;5;143m'      # Soft olive green
WARNING='\033[38;5;29m'      # Forest green
SECONDARY='\033[38;5;214m'     # Warm orange
ERROR='\033[38;5;167m'       # Soft red
INFO='\033[38;5;109m'        # Dusty blue
SUBTLE='\033[38;5;245m'      # Medium gray
BRIGHT='\033[38;5;255m'      # Clean white
DIM_TEXT='\033[38;5;240m'    # Dark gray
HIGHLIGHT='\033[38;5;180m'   # Warm beige
NEUTRAL='\033[38;5;252m'     # Light gray
NC='\033[0m' # No Color

# Text formatting - Enhanced
BOLD='\033[1m'
UNDERLINE='\033[4m'
DIM='\033[2m'
ITALIC='\033[3m'
BLINK='\033[5m'
REVERSE='\033[7m'
STRIKETHROUGH='\033[9m'

# Background colors
BG_BLACK='\033[40m'
BG_RED='\033[41m'
BG_GREEN='\033[42m'
BG_YELLOW='\033[43m'
BG_BLUE='\033[44m'
BG_PURPLE='\033[45m'
BG_CYAN='\033[46m'
BG_WHITE='\033[47m'


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

# Clear screen for better presentation
clear

# ASCII Art Header
echo -e "\n"
echo -e "${PRIMARY}${BOLD}     ██████╗ ██████╗ ██╗   ██╗    ${NC}"
echo -e "${PRIMARY}${BOLD}    ██╔════╝ ██╔══██╗██║   ██║    ${NC}"
echo -e "${PRIMARY}${BOLD}    ██║  ███╗██████╔╝██║   ██║    ${NC}"
echo -e "${PRIMARY}${BOLD}    ██║   ██║██╔══██╗██║   ██║    ${NC}"
echo -e "${PRIMARY}${BOLD}    ╚██████╔╝██║  ██║╚██████╔╝    ${NC}"
echo -e "${PRIMARY}${BOLD}     ╚═════╝ ╚═╝  ╚═╝ ╚═════╝     ${NC}"
echo -e "\n"
echo -e "${SECONDARY}${BOLD}████████╗██████╗  █████╗ ██████╗ ███████╗    ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗${NC}"
echo -e "${SECONDARY}${BOLD}╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝${NC}"
echo -e "${SECONDARY}${BOLD}   ██║   ██████╔╝███████║██║  ██║█████╗      █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  ${NC}"
echo -e "${SECONDARY}${BOLD}   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝      ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  ${NC}"
echo -e "${SECONDARY}${BOLD}   ██║   ██║  ██║██║  ██║██████╔╝███████╗    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗${NC}"
echo -e "${SECONDARY}${BOLD}   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝    ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝${NC}"
echo -e "\n"
echo -e "${DIM_TEXT}                                    Vroom vroom mudder trucker                                       ${NC}"
echo -e "\n"

read -p "Would you like to install/update(y/n) [type 'r' to force reinstall]: " choice


if [[ "$choice" == "r" || "$choice" == "R" ]]; then
        echo -e "${WARNING}Reinstalling...${NC}"
        yes | rm -r .venv
        rm -r algorithms/sentiment_model/web_scraper/build
        mkdir algorithms/sentiment_model/web_scraper/build
        show_progress 2 "Removed existing .venv directory."
        choice="y"
fi

mkdir data

if [[ "$choice" == "y" || "$choice" == "Y" ]]; then

    show_progress 2 "Checking dependencies."

    sudo apt update && sudo apt upgrade -y 

    sudo apt install -y python3-full  

    python3 -m venv .venv 

    source .venv/bin/activate
    pip install -r algorithms/requirements.txt

    sudo apt install -y libcurl4-openssl-dev 
    sudo apt install -y libfmt-dev 

    home=$(pwd)

    yes| rm -r algorithms/sentiment_model/web_scraper/build/CMakeCache.txt

    cd algorithms/sentiment_model/web_scraper/build
    show_progress 2 "Building c++."
    cmake ..
    make
    chmod +x webscrape.exe

    cd $home
else
    echo -e "${INFO}Skipping installation...${NC}"
    source .venv/bin/activate
fi

source .venv/bin/activate

# === Custom Training Workflow ===

read -p "Enter stock ticker (e.g., AAPL): " stock_ticker
read -p "Enter the name of this company: " company_name

read -p "Train sentiment analysis model? (y/n): " train_sentiment
read -p "Train GRU model? (y/n): " train_gru

if [[ "$train_sentiment" == "y" || "$train_sentiment" == "Y" ]]; then
    echo -e "${INFO}Training sentiment analysis model...${NC}"
    "$home"/algorithms/sentiment_model/web_scraper/build/webscrape.exe data "$company_name"
    python3 main.py --mode s --symbol "$stock_ticker"
fi

if [[ "$train_gru" == "y" || "$train_gru" == "Y" ]]; then
    echo -e "${INFO}Training GRU model...${NC}"
    python3 main.py --mode t --symbol "$stock_ticker" --epochs 30 --batch_size 10 --lr 0.001
fi

echo -e "${SUCCESS}Training workflow complete!${NC}"

