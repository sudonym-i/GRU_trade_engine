import engine.tsr_model 
import engine.sentiment_model


def anylize_stock(ticker: str):
    model = engine.tsr_model.train_model()
    #engine.tsr_model.test_run_model(model)


# to be wrapped into REST api

def main():
    ticker = "AAPL"

    # Tracking a specific stock will begin with a POST request

    # updates to information can be acquired thorugh GET requests,
    # which returns data states from api that are updating on a schedule
    # that keeps me below the api limit

    # POST requests be used to terminate tracking


    anylize_stock(ticker)


main()