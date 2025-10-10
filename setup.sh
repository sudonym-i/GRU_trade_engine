
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
mkdir algorithms/gru_model/models

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

# === Configuration Setup ===

echo -e "\n${PRIMARY}${BOLD}Configuration Setup${NC}"
echo -e "${INFO}Setting up config.json for your stock...${NC}\n"

read -p "Enter stock ticker (e.g., AAPL): " stock_ticker
read -p "Enter the name of this company: " company_name

echo -e "\n${PRIMARY}${BOLD}Training Mode Selection${NC}"
echo -e "${INFO}Choose training approach:${NC}"
echo -e "  ${HIGHLIGHT}1)${NC} Standard single-stock training (traditional)"
echo -e "  ${HIGHLIGHT}2)${NC} Transfer learning (pre-train on multiple stocks, then fine-tune)"
echo -e ""
read -p "Select mode (1 or 2, default 1): " training_mode
training_mode=${training_mode:-1}

if [[ "$training_mode" == "2" ]]; then
    echo -e "\n${PRIMARY}${BOLD}Transfer Learning Configuration${NC}"
    echo -e "${INFO}Enter stocks for pre-training (comma-separated, e.g., AMD,NVDA,INTC,TSM):${NC}"
    read -p "Pre-training stocks: " pretrain_stocks_input
    pretrain_stocks_input=${pretrain_stocks_input:-"AMD,NVDA,INTC,TSM,MU,QCOM,AVGO"}

    read -p "Pre-training epochs (default 40): " pretrain_epochs
    pretrain_epochs=${pretrain_epochs:-40}

    read -p "Pre-training learning rate (default 0.0005): " pretrain_lr
    pretrain_lr=${pretrain_lr:-0.0005}

    read -p "Fine-tuning epochs (default 25): " finetune_epochs
    finetune_epochs=${finetune_epochs:-25}

    read -p "Fine-tuning learning rate (default 0.0001): " finetune_lr
    finetune_lr=${finetune_lr:-0.0001}

    read -p "Batch size (default 32): " batch_size
    batch_size=${batch_size:-32}

    epochs=$finetune_epochs
    learning_rate=$finetune_lr
else
    echo -e "\n${PRIMARY}${BOLD}Standard Training Configuration${NC}"
    read -p "Training epochs (default 50): " epochs
    epochs=${epochs:-50}

    read -p "Learning rate (default 0.0005): " learning_rate
    learning_rate=${learning_rate:-0.0005}

    read -p "Batch size (default 32): " batch_size
    batch_size=${batch_size:-32}
fi

# Update config.json with user inputs
if [[ "$training_mode" == "2" ]]; then
    # Transfer learning mode
    python3 << EOF
import json

# Load existing config or create new one
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    # Use example config as template
    with open('config.example.json', 'r') as f:
        config = json.load(f)

# Ensure training section exists with all required fields
if 'training' not in config:
    config['training'] = {}

# Convert pretrain stocks string to list
pretrain_stocks_str = '$pretrain_stocks_input'
pretrain_stocks = [s.strip() for s in pretrain_stocks_str.split(',')]

# Update with user inputs
config['stock']['ticker'] = '$stock_ticker'
config['stock']['company_name'] = '$company_name'
config['training']['mode'] = 'transfer_learning'
config['training']['epochs'] = int($epochs)
config['training']['learning_rate'] = float($learning_rate)
config['training']['batch_size'] = int($batch_size)
config['training']['pretrain_stocks'] = pretrain_stocks
config['training']['pretrain_epochs'] = int($pretrain_epochs)
config['training']['pretrain_lr'] = float($pretrain_lr)
config['training']['finetune_epochs'] = int($finetune_epochs)
config['training']['finetune_lr'] = float($finetune_lr)
config['training']['validation_split'] = 0.2
config['training']['pretrained_model_path'] = 'algorithms/gru_model/models/pretrained_gru.pth'

# Preserve or add loss_function configuration
if 'loss_function' not in config['training']:
    config['training']['loss_function'] = {
        'type': 'directional',
        'mse_weight': 0.3,
        'direction_weight': 0.5,
        'bias_weight': 0.2,
        'direction_penalty': 2.0
    }

# Save updated config
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Transfer Learning configuration saved to config.json")
print(f"  Stock: $stock_ticker ($company_name)")
print(f"  Pre-training: {len(pretrain_stocks)} stocks, epochs={$pretrain_epochs}, lr={$pretrain_lr}")
print(f"  Fine-tuning: epochs={$finetune_epochs}, lr={$finetune_lr}, batch_size={$batch_size}")
EOF
else
    # Standard training mode
    python3 << EOF
import json

# Load existing config or create new one
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    # Use example config as template
    with open('config.example.json', 'r') as f:
        config = json.load(f)

# Update with user inputs
config['stock']['ticker'] = '$stock_ticker'
config['stock']['company_name'] = '$company_name'
config['training']['mode'] = 'standard'
config['training']['epochs'] = int($epochs)
config['training']['learning_rate'] = float($learning_rate)
config['training']['batch_size'] = int($batch_size)

# Preserve or add loss_function configuration
if 'loss_function' not in config['training']:
    config['training']['loss_function'] = {
        'type': 'directional',
        'mse_weight': 0.3,
        'direction_weight': 0.5,
        'bias_weight': 0.2,
        'direction_penalty': 2.0
    }

# Save updated config
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Standard training configuration saved to config.json")
print(f"  Stock: $stock_ticker ($company_name)")
print(f"  Training: epochs={$epochs}, lr={$learning_rate}, batch_size={$batch_size}")
EOF
fi

echo ""

# === Training Workflow ===

read -p "Train sentiment analysis model? (y/n): " train_sentiment
read -p "Train GRU model? (y/n): " train_gru

if [[ "$train_sentiment" == "y" || "$train_sentiment" == "Y" ]]; then
    echo -e "${INFO}Training sentiment analysis model...${NC}"
    "$home"/algorithms/sentiment_model/web_scraper/build/webscrape.exe data "$company_name"
    python3 main.py --mode s
fi

if [[ "$train_gru" == "y" || "$train_gru" == "Y" ]]; then
    if [[ "$training_mode" == "2" ]]; then
        echo -e "\n${PRIMARY}${BOLD}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${PRIMARY}${BOLD}         TRANSFER LEARNING: TWO-PHASE TRAINING${NC}"
        echo -e "${PRIMARY}${BOLD}═══════════════════════════════════════════════════════════${NC}\n"

        echo -e "${INFO}Phase 1: Pre-training on multiple stocks...${NC}"
        echo -e "${SUBTLE}Stocks: $pretrain_stocks_input${NC}"
        echo -e "${SUBTLE}Epochs: $pretrain_epochs, LR: $pretrain_lr, Batch: $batch_size${NC}\n"
        python3 main.py --mode pretrain

        echo -e "\n${INFO}Phase 2: Fine-tuning on target stock ($stock_ticker)...${NC}"
        echo -e "${SUBTLE}Epochs: $finetune_epochs, LR: $finetune_lr, Batch: $batch_size${NC}\n"
        python3 main.py --mode finetune

        echo -e "\n${SUCCESS}${BOLD}✓ Transfer learning complete!${NC}"
        echo -e "${INFO}Pre-trained model: algorithms/gru_model/models/pretrained_gru.pth${NC}"
        echo -e "${INFO}Fine-tuned model: algorithms/gru_model/models/gru_model.pth${NC}\n"
    else
        echo -e "\n${INFO}Training GRU model (standard single-stock)...${NC}"
        echo -e "${SUBTLE}Using config: epochs=$epochs, lr=$learning_rate, batch_size=$batch_size${NC}\n"
        python3 main.py --mode t

        echo -e "\n${SUCCESS}${BOLD}✓ Training complete!${NC}"
        echo -e "${INFO}Model saved: algorithms/gru_model/models/gru_model.pth${NC}\n"
    fi
fi

echo -e "\n${PRIMARY}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${SUCCESS}${BOLD}           Setup workflow complete!${NC}"
echo -e "${PRIMARY}${BOLD}═══════════════════════════════════════════════════════════${NC}\n"

if [[ "$training_mode" == "2" ]]; then
    echo -e "${INFO}Next steps:${NC}"
    echo -e "  • Make predictions: ${HIGHLIGHT}python3 main.py --mode p${NC}"
    echo -e "  • View transfer learning guide: ${HIGHLIGHT}cat TRANSFER_LEARNING_GUIDE.md${NC}"
else
    echo -e "${INFO}Next steps:${NC}"
    echo -e "  • Make predictions: ${HIGHLIGHT}python3 main.py --mode p${NC}"
    echo -e "  • Try transfer learning: ${HIGHLIGHT}./setup.sh${NC} (select option 2)"
fi
echo -e ""

