import tensorflow as tf
class preprocessing():
  def __init__(self):
    pass
  def text2tf_dataset(self,text,target_data,target_input_data,MAX_VOCAB_SIZE=10000,filters='',padding='post',BATCH_SIZE=32):   
    # input     
    tokenizer_inputs = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE, filters=filters)
    tokenizer_inputs.fit_on_texts(text)
    input_sequences = tokenizer_inputs.texts_to_sequences(text)
    input_max_len = max(len(s) for s in input_sequences)
    # output
    tokenizer_outputs = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE, filters=filters)
    tokenizer_outputs.fit_on_texts(target_data)
    tokenizer_outputs.fit_on_texts(target_input_data)
    target_data = tokenizer_outputs.texts_to_sequences(target_data)
    target_input_data = tokenizer_outputs.texts_to_sequences(target_input_data)
    target_max_len = max(len(s) for s in target_data)
    # Vocab
    word2idx_inputs = tokenizer_inputs.word_index
    print('Found %s unique input tokens.' % len(word2idx_inputs))
    # get the word to index mapping for output language
    word2idx_outputs = tokenizer_outputs.word_index
    print('Found %s unique output tokens.' % len(word2idx_outputs))
    num_words_output = len(word2idx_outputs) + 1
    num_words_inputs = len(word2idx_inputs) + 1
    idx2word_inputs = {v:k for k, v in word2idx_inputs.items()}
    idx2word_outputs = {v:k for k, v in word2idx_outputs.items()}
    # Padding 
    encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=input_max_len, padding=padding)
    print("encoder_inputs.shape:", encoder_inputs.shape)
    print("encoder_inputs[0]:", encoder_inputs[0])
    # pad the decoder input sequences
    decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(target_input_data, maxlen=target_max_len, padding=padding)
    print("decoder_inputs[0]:", decoder_inputs[0])
    print("decoder_inputs.shape:", decoder_inputs.shape)
    # pad the target output sequences
    decoder_targets = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=target_max_len, padding=padding)
    # create a Dataset using the tf.data
    dataset = tf.data.Dataset.from_tensor_slices(
    (encoder_inputs, decoder_inputs, decoder_targets))
    dataset = dataset.shuffle(len(input_sequences)).batch(
        BATCH_SIZE, drop_remainder=True)
    
    return dataset,idx2word_inputs,word2idx_outputs

input_data_path='/content/drive/MyDrive/Colab Notebooks/Naive_NMT/dataset/input_data.txt'
target_data_path='/content/drive/MyDrive/Colab Notebooks/Naive_NMT/dataset/target_data.txt'
target_input_data_path='/content/drive/MyDrive/Colab Notebooks/Naive_NMT/dataset/target_input_data.txt'
with open(input_data_path) as f:
    input_data = f.read().splitlines()
with open(target_data_path) as f:
    target_data = f.read().splitlines()
with open(target_input_data_path) as f:
    target_input_data = f.read().splitlines()

"""
text=["Create the input and output vocabularies: Using the tokenizer we’ve created",
      "For a better understanding, we can divide the model in three basic components"]
p=preprocessing()
data,idx2word_inputs,word2idx_outputs=p.text2tf_dataset(text,text,text)
"""