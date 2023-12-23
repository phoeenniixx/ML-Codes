# **Neural Machine Translation**

1. Neural Machine Translation (NMT) is the task of using artificial neural network models for translation from one language to the other.
2. The NMT model generally consists of an encoder that encodes a source sentence into a fixed-length vector from which a decoder generates a translation.
3. This problem can be thought as a prediction problem, where given a sequence of words in source language as input, task is to predict the output sequence of words in target language.
4. The dataset comes from http://www.manythings.org/anki/, where you may find tab delimited bilingual sentence pairs in different files based on the source and target language of your choice

The project involves building a machine translation model using a sequence-to-sequence (seq2seq) architecture with an encoder-decoder framework. The goal is to translate sentences from one language to another. The project can be broken down into several key steps:

Step 1: Download and Clean the Data
1. Download a dataset containing language pairs (source and target phrases).
2. Extract the data and read it into a format suitable for processing.
3. Clean the data by removing non-printable characters, punctuation, and non-alphabetic characters.
4. Convert the text to lowercase for uniformity.

Step 2: Split and Prepare the Data for Training
1. Split the cleaned data into training and testing sets.
2. Create separate tokenizers for the source and target languages to convert text into numerical sequences.
3. Encode and pad the input sequences (source language) and output sequences (target language) based on the individual tokenizers and maximum sequence lengths.
4. Perform one-hot encoding on the output sequences, as the goal is to predict words in the target language.
   
Step 3: Define and Train the RNN-based Encoder-Decoder Model
1. Define a sequential model with two main parts: Encoder and Decoder.
2. In the Encoder, pass the input sequence through an Embedding layer to train word embeddings for the source language. Additionally, use one or more RNN/LSTM layers to capture sequential information.
3. Connect the Encoder to the Decoder using a Repeat Vector layer to match the shapes of the output from the Encoder and the expected input by the Decoder.
4. In the Decoder, stack one or more RNN/LSTM layers and add a Time Distributed Dense layer to produce outputs for each timestep.

   
The model is trained to learn the mapping between the source language and target language, allowing it to generate translations for new input sentences. The training process involves optimizing the model's parameters to minimize the difference between predicted and actual translations. The effectiveness of the model can be evaluated on the test set, and adjustments can be made to improve its performance if necessary.





