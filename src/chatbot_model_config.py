
def config():
    # Configure models

    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.25
    batch_size = 64

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0

    MAX_LENGTH = 12  # Maximum sentence length to consider
    MIN_COUNT = 2  # Minimum word count threshold for trimming

    return hidden_size, encoder_n_layers, decoder_n_layers, dropout, batch_size, learning_rate, decoder_learning_ratio, \
           teacher_forcing_ratio, clip, MAX_LENGTH,  MIN_COUNT

