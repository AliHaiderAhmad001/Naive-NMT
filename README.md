# Naive-model for Neural Machine Translation(NMT)
"Machine translation (MT) is the task of automatically converting source text in one language to text in another language. Given a sequence of text in a source language, there is no one single best translation of that text to another language. This is because of the natural ambiguity and flexibility of human language. This makes the challenge of automatic machine translation difficult, perhaps one of the most difficult in artificial intelligence."

Machine Learning Mastery, Jason Brownlee

------------------------------------------------
The seq2seq model consists of two sub-networks, the encoder and the decoder. The encoder, on the left hand, receives sequences from the source language as inputs and produces as a result a compact representation of the input sequence, trying to summarize or condense all its information. Then that output becomes an input or initial state of the decoder, which can also receive another external input. At each time step, the decoder generates an element of its output sequence based on the input received and its current state, as well as updating its own state for the next time step. 
![Encode-Decoder](https://blog.keras.io/img/seq2seq/seq2seq-teacher-forcing.png)

## Dataset
This is the [link](http://www.manythings.org/anki/) to download the Spanish-English spa_eng.zip file.
