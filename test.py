import sys
sys.path.append('./algorithms/gru_model')

from algorithms.gru_model.gru_object import GRUModel



# ============================
# sample data
symbol = 'NVDA'
input_size = 5  # OHLCV
hidden_size = 32
output_size = 1
sequence_length = 30
# ============================


def main():

    # initialize a gru model object
    tsr_model = GRUModel(input_size, hidden_size, output_size)

    # specify where the collected data ought to be stored
    tsr_model.data_dir = "./data"

    # tell the object to pull neccesary data for itself
    data = tsr_model.pull_data(symbol=symbol)

    # this tells the object to format and normalize the data for itself
    data = tsr_model.format_data(data)

    # trains its GRU time series prediction
    tsr_model.train(data, epochs=5, lr=0.001)

    # test- printing its prediction (still normalized right now)
    print(tsr_model.predict(data))

#testing
if __name__ == "__main__":
    main()