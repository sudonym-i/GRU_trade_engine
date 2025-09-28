
source .venv/bin/activate

# Read parameters from params.txt
if [ -f data/params.txt ]; then
    stock_ticker=$(grep '^stock_ticker=' data/params.txt | cut -d'=' -f2)
    company_name=$(grep '^company_name=' data/params.txt | cut -d'=' -f2)
else
    echo "params.txt not found!" >&2
    exit 1
fi

# 1. Run GRU prediction
python3 main.py --mode p --symbol "$stock_ticker" > data/output/gru_prediction.out

# 2. Run webscraper (C++ program)
"$(pwd)"/algorithms/sentiment_model/web_scraper/build/webscrape.exe data "$company_name"

# 3. Run sentiment analysis
python3 main.py --mode s --symbol "$stock_ticker" > data/output/sentiment_analysis.out

# 4. Send results to Discord
python3 main.py --mode discord --symbol "$stock_ticker" --prediction_file data/output/gru_prediction.out --sentiment_file data/output/sentiment_analysis.out

echo -e "${SUCCESS}Runthrough complete!${NC}"


