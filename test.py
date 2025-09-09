import sys
sys.path.append('./algorithms/gru_model')

from algorithms.gru_model.gru_object import GRUModel

def main():

    # ============================
    # sample data
    symbol = 'NVDA'
    input_size = 5  # OHLCV
    hidden_size = 32
    output_size = 1
    sequence_length = 30
    # ============================

    # Pull data
    tsr_model = GRUModel(input_size, hidden_size, output_size)

    data = tsr_model.pull_data(symbol=symbol)
    data = tsr_model.format_data(data)
    tsr_model.train(data, epochs=5, lr=0.001)
    print(tsr_model.predict(data))

if __name__ == "__main__":
    main()