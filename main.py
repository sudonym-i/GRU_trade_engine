import sys
sys.path.append('./algorithms/gru_model')

from algorithms.gru_model.gru_object import GRUModel



# ============================
# sample data
symbol = 'NVDA'
input_size = 5  # OHLCV
hidden_size = 256
output_size = 1
sequence_length = 30
# ============================


def main():

    mode = input("Train or predict? (t/p) ").strip().lower()


    if mode == 't':
    # # ---------- TRAIN MODEL -------------
        # initialize a gru model object
        gru_model = GRUModel(input_size, hidden_size, output_size)

        # specify where the collected data ought to be stored
        gru_model.data_dir = "./data"

        # tell the object to pull neccesary data for itself
        gru_model.pull_data(symbol=symbol, period="max")

        # this tells the object to format and normalize the data for both training and criterion
        # "scaler" is returned for un-normalizing predictions later
        gru_model.format_data()

        # trains its GRU time series prediction
        gru_model.train( epochs=30, lr=0.001 )

        gru_model.save_model( f"./algorithms/gru_model/models/{symbol}_gru_model.pth" )


# -------------- TEST MODEL ------------
    else:
        gru_model = GRUModel(input_size, hidden_size, output_size)

        gru_model.data_dir = "./data"

        gru_model.load_model( f"./algorithms/gru_model/models/{symbol}_gru_model.pth" ) 

        gru_model.pull_data(symbol=symbol, period="1y")
        gru_model.format_data()
        gru_model.predict()

        price_prediction = gru_model.un_normalize()

        print(price_prediction)

#testing
if __name__ == "__main__":
    main()