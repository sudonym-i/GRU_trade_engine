
#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Text formatting
BOLD='\033[1m'
UNDERLINE='\033[4m'
DIM='\033[2m'

# Clear screen for better presentation
clear

echo -e "${CYAN}${BOLD}╔════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║           Neural Trade Engine          ║${NC}"
echo -e "${CYAN}${BOLD}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GRAY}Initializing automated trading system...${NC}"
echo ""

# Save current directory, hopefully help with the path issues
here=$(pwd)

# May need to automate model downloading as well

echo -e "${YELLOW}${BOLD}[STEP 1/4]${NC} ${BLUE}Setting up Python virtual environment...${NC}"
echo -e "${GRAY}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
sudo apt install python3-full

if [ ! -d "$here/.venv" ]; then
    python3 -m venv "$here/.venv"
fi

source "$here/.venv/bin/activate"

echo -e "\n${GREEN}✓${NC} Virtual environment ready"
echo -e "${YELLOW}${BOLD}[STEP 2/4]${NC} ${BLUE}Installing Python dependencies...${NC}"
echo -e "${GRAY}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
pip install -r "$here/backend_&_algorithms/engine/requirements.txt"
echo -e "\n${GREEN}✓${NC} Python dependencies installed"
echo -e "\n${YELLOW}${BOLD}[STEP 3/4]${NC} ${BLUE}Setting up C++ environment...${NC}"
echo -e "${GRAY}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
mkdir "$here/backend_&_algorithms/engine/sentiment_model/web_scraper/build"
sudo apt-get update && sudo apt upgrade
sudo apt install cmake
sudo apt-get install libgtest-dev
sudo apt-get install cmake
sudo apt install libfmt-dev 
sudo apt install curl
sudo apt install libcurl4-gnutls-dev
echo -e "\n${PURPLE}Compiling C++ components...${NC}"
cd "$here/backend_&_algorithms/engine/sentiment_model/web_scraper/build"
cmake ..
make
chmod +x webscrape.exe && echo -e "${GREEN}✓ C++ build successful${NC}"
cd "$here"

echo -e "\n${YELLOW}${BOLD}[STEP 4/4]${NC} ${GREEN}${BOLD}Setup complete!${NC}"
echo -e "${CYAN}${BOLD}╔════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║          Configuration Phase           ║${NC}"
echo -e "${CYAN}${BOLD}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}❯${NC} What ticker do you want to algorithmically trade?"
read -p "  Enter ticker symbol: " ticker
echo -e "${YELLOW}❯${NC} Enter semantic name (EX: GOOG would be -> google)"
read -p "  Semantic name: " semantic_name

echo -e "\n${YELLOW}❯${NC} ${WHITE}Select time interval for trading:${NC}"
echo -e "  ${CYAN}1${NC} → 1hr"
echo -e "  ${CYAN}2${NC} → 5min"
echo -e "  ${CYAN}3${NC} → 15min"
echo -e "  ${CYAN}4${NC} → 30min"
echo -e "  ${CYAN}5${NC} → 1d"
read -p "  Enter your choice (1-5): " time_interval

# Convert numeric choice to actual time interval
case $time_interval in
    1) interval_value="1hr" ;;
    2) interval_value="5min" ;;
    3) interval_value="15min" ;;
    4) interval_value="30min" ;;
    5) interval_value="1d" ;;
    *) interval_value="1hr" ;; # default
esac

echo -e "\n${YELLOW}❯${NC} ${WHITE}Train sentiment model?${NC}"
echo -e "  ${GRAY}(Recommended for better accuracy)${NC}"
read -p "  Train model? (y/n): " train_models
echo ""


# if these directories don't exist, create them
if [ ! -d "$here/backend_&_algorithms/engine/sentiment_model/raw_data" ]; then
    mkdir "$here/backend_&_algorithms/engine/sentiment_model/raw_data"
fi

if [ ! -d "$here/backend_&_algorithms/engine/sentiment_model/processed_data" ]; then
    mkdir "$here/backend_&_algorithms/engine/sentiment_model/processed_data"
fi

echo -e "\n${BLUE}${BOLD}Data Collection${NC}"
echo -e "${GRAY}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PURPLE}Scraping web data for ${CYAN}$semantic_name${PURPLE}...${NC}"
cd "$here/backend_&_algorithms/"
"$here/.venv/bin/python3" "main.py" webscrape --ticker "$semantic_name"
echo -e "${GREEN}✓ Web scraping complete${NC}"

if [ "$train_models" == "y" ] || [ "$train_models" == "Y" ]; then
    echo -e "${BLUE}${BOLD}Training Sentiment Model${NC}"
    echo -e "${GRAY}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    cd "$here/backend_&_algorithms/engine/sentiment_model/"
    echo -e "${PURPLE}Downloading dataset...${NC}"
    "$here/.venv/bin/python3" "$here/backend_&_algorithms/engine/sentiment_model/download_dataset.py"
    echo -e "${PURPLE}Tokenizing data...${NC}"
    "$here/.venv/bin/python3" "$here/backend_&_algorithms/engine/sentiment_model/tokenize_pipeline.py"
    echo -e "${PURPLE}Training model (this may take a while)...${NC}"
    "$here/.venv/bin/python3" "$here/backend_&_algorithms/engine/sentiment_model/train.py"
    echo -e "${GREEN}✓ Sentiment model training complete${NC}"
fi

cd "$here"


echo -e "\n${YELLOW}❯${NC} ${WHITE}Train TSR model?${NC}"
echo -e "  ${GRAY}(Required for trading functionality)${NC}"
read -p "  Train TSR model? (y/n): " train_tsr
echo ""

cd "$here/backend_&_algorithms/"
if [ "$train_tsr" == "y" ] || [ "$train_tsr" == "Y" ]; then
    echo -e "${BLUE}${BOLD}Training TSR Model${NC}"
    echo -e "${GRAY}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}Training TSR model for ${CYAN}$ticker${PURPLE}...${NC}"
    "$here/.venv/bin/python3" main.py train --ticker "$ticker"
    echo -e "${GREEN}✓ TSR model training complete${NC}"
fi

cd $here/integrations_\&_strategy
# Virtual environment activated via direct python path

echo -e "\n${CYAN}${BOLD}╔════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║            Trading Setup               ║${NC}"
echo -e "${CYAN}${BOLD}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}❯${NC} ${WHITE}Start automated trading now?${NC}"
read -p "  Start trading? (y/n): " start
if [ "$start" == "y" ] || [ "$start" == "Y" ]; then
    echo ""
    echo -e "${WHITE}${BOLD}Select Trading Mode:${NC}"
    echo -e "  ${GREEN}1${NC} → ${GREEN}Simulation Mode${NC} ${GRAY}(paper trading - recommended for testing)${NC}"
    echo -e "  ${YELLOW}2${NC} → ${YELLOW}IB Paper Trading${NC} ${GRAY}(requires IB Gateway/TWS)${NC}"
    echo -e "  ${RED}3${NC} → ${RED}IB Live Trading${NC} ${GRAY}(real money - use with caution!)${NC}"
    echo ""
    read -p "  Enter your choice (1-3): " mode_choice
    
    case $mode_choice in
        1)
            echo -e "\n${GREEN}${BOLD}▶ Starting SIMULATION mode...${NC}"
            echo -e "  ${GRAY}Portfolio will use virtual \$10,000 starting capital${NC}"
            # Update config.json with ticker, semantic_name, mode and time_interval
            jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "simulation" --arg interval "$interval_value" \
                '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                config.json > config.json.tmp && mv config.json.tmp config.json
            nano config.json
            $here/.venv/bin/python3 schedule_trader.py --start
            ;;
        2)
            echo -e "\n${YELLOW}${BOLD}▶ Starting IB PAPER TRADING mode...${NC}"
            echo -e "  ${BLUE}Testing IB connection first...${NC}"
            # Update config.json with ticker, semantic_name, mode and time_interval
            jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "ib_paper" --arg interval "$interval_value" \
                '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                config.json > config.json.tmp && mv config.json.tmp config.json
            $here/.venv/bin/python3 test_ib_connection.py --mode paper
            if [ $? -eq 0 ]; then
                $here/.venv/bin/python3 schedule_trader.py --start
            else
                echo -e "\n${RED}${BOLD}✗ IB connection test failed${NC}"
                echo -e "  ${YELLOW}Please check:${NC}"
                echo -e "    ${GRAY}• IB Gateway or TWS is running${NC}"
                echo -e "    ${GRAY}• API is enabled in IB settings${NC}"
                echo -e "    ${GRAY}• Paper trading port 7496 is configured${NC}"
                echo -e "\n${YELLOW}${BOLD}↻ Falling back to simulation mode...${NC}"
                # Update config back to simulation mode
                jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "simulation" --arg interval "$interval_value" \
                    '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                    config.json > config.json.tmp && mv config.json.tmp config.json
                nano config.json
                $here/.venv/bin/python3 schedule_trader.py --start
            fi
            ;;
        3)
            echo -e "\n${RED}${BOLD}⚠️  WARNING: LIVE TRADING MODE${NC}"
            echo -e "  ${RED}This will execute actual trades with real money!${NC}"
            echo -e "  ${YELLOW}Trading will occur in your Interactive Brokers account${NC}"
            echo ""
            read -p "  Are you absolutely sure you want to continue? (yes/no): " confirm
            if [ "$confirm" == "yes" ]; then
                echo -e "\n${BLUE}Testing IB live connection first...${NC}"
                # Update config.json with ticker, semantic_name, mode and time_interval
                jq --arg ticker "$ticker" --arg semantic_name "$semantic_name" --arg mode "ib_live" --arg interval "$interval_value" \
                    '.target_stock = $ticker | .semantic_name = $semantic_name | .trading_mode = $mode | .time_interval = $interval' \
                    config.json > config.json.tmp && mv config.json.tmp config.json
                nano config.json
                $here/.venv/bin/python3 test_ib_connection.py --mode live
                if [ $? -eq 0 ]; then
                    echo -e "\n${GREEN}${BOLD}🚀 Starting LIVE TRADING - Good luck!${NC}"
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
