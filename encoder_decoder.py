import tensorflow as tf
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # LSTM
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True)

    def call(self, input_sequence, states):
        # Embed the input
        embed = self.embedding(input_sequence)
        # Call the LSTM unit
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size):
        # Return a all 0s initial states
        return (tf.zeros([batch_size, self.hidden_dim]),
                tf.zeros([batch_size, self.hidden_dim]))

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"hidden_dim":self.hidden_dim,
                       "embedding": self.embedding,
                        "lstm":self.lstm})
        return config

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Define the embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # Define the RNN layer, LSTM
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, input_sequence, state):
        # Embed the input
        embed = self.embedding(input_sequence)
        # Call the LSTM unit
        lstm_out, state_h, state_c = self.lstm(embed, state)
        # Dense layer to predict output token
        logits = self.dense(lstm_out)
        return logits, state_h, state_c

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"hidden_dim":self.hidden_dim,
                       "embedding": self.embedding,
                       "lstm":self.lstm,
                       "dense":self.dense})
        return config