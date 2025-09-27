
    # 1. Run GRU prediction
python3 main.py --mode p --symbol "$stock_ticker" > gru_prediction.out

    # 2. Run webscraper (C++ program)
"$home"/algorithms/sentiment_model/web_scraper/build/webscrape.exe data "$semantic_name"

    # 3. Run sentiment analysis (assuming main.py or another script)
python3 main.py --mode s --symbol "$stock_ticker" > sentiment_analysis.out

    # 4. Send results to Discord using main.py (add a CLI option for this if needed)
python3 main.py --mode discord --symbol "$stock_ticker" --prediction_file gru_prediction.out --sentiment_file sentiment_analysis.out


echo -e "${SUCCESS}Runthrough complete!${NC}"


