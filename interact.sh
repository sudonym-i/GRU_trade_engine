
#!/bin/bash

# Color definitions 
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

