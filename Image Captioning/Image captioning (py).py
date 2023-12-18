# Import Libraries
import os
import scipy.io
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.models import Model

from keras.preprocessing.image import load_img
from keras.layers import Input, Dense, Embedding, LSTM, concatenate, Dropout
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Add
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
import string
import tensorflow as tf
import keras
from nltk.tokenize import word_tokenize

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model

from pickle import dump

from google.colab import drive

drive.mount('/content/drive')
# Image Feature Extraction
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)# summarize
print(model.summary())

img=load_img('/content/gdrive/MyDrive/archive/Images/3767841911_6678052eb6.jpg',target_size=(224, 224))


def img_preprocess(filename):
  feats={}
  j=0
  for img in os.listdir(filename):
    image=load_img(f'{filename}/{img}',target_size=(224, 224))
    img_arr=img_to_array(image)
    img_arr = img_arr.reshape((1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
    feat=model.predict(img_arr,verbose=0)#no output shown as verbose=0
    img_id=img.split('.')[0]
    feats[img_id]=feat
    print(j)
    j+=1
  return feats

 features=img_preprocess('/content/gdrive/MyDrive/archive/Images/')

pkl_file_path = '/content/drive/MyDrive/archive/your_data.pkl'

import pickle
pkl_file_path = '/content/drive/MyDrive/archive/your_data.pkl'
with open(pkl_file_path, 'wb') as pkl_file:
   pickle.dump(features, pkl_file)

from pickle import load
all_features = load(open(pkl_file_path, 'rb'))

len(list(all_features.values()))

# Text Loading and Cleaning

file_path='/content/drive/MyDrive/archive/captions.txt'
with open(file_path, 'r') as file:
    content = file.read()


s = set(string.punctuation)

def clean(caption):
    cleaned = []
    x="startseq"
    for word in word_tokenize(caption):
        if (word.lower() not in s):
            cleaned.append(word.lower())
    for i in cleaned:
        if i==cleaned[len(cleaned)-1]:
            x=x+" "+i+" endseq"
        else:
            x=x+" "+ i
    return x

def create_data(content):
    captions = {}
    for line in content.split('\n'):
        if line == 'image,caption' or line == '':
            continue
        cap = line.split(',')[1:]
        im = str(line.split(',')[0])
        img_id = str(im.split('.')[0])
        if img_id not in captions:
            captions[img_id] = []
        caps = clean(' '.join(cap))
        captions[img_id].append(caps)
    return captions

cap = create_data(content)


def to_vocabulary(descriptions):
 all_desc = set()
 for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
 return all_desc

vocab=list(to_vocabulary(cap))


import pickle
pkl_file_path = '/content/drive/MyDrive/archive/vocab.pkl'
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(vocab, pkl_file)

def to_lines(descriptions):
 all_desc = []
 for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]
 return all_desc

# Tokenizer and creating train and test data

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer



def max_length(descriptions):
  lines = to_lines(descriptions)
  return max(len(d.split()) for d in lines)

def train_test_split(desc, feats):
    train_desc = {k: v for i, (k, v) in enumerate(desc.items()) if i < 6000}
    test_desc = {k: v for i, (k, v) in enumerate(desc.items()) if i >= 6000}
    selected_features = {key: all_features[key] for key in train_desc.keys() & all_features.keys()}
    train_feats = {key: selected_features[key] for key in train_desc.keys()}

    test_features = {key: all_features[key] for key in test_desc.keys() & all_features.keys()}
    test_feats= {key: test_features[key] for key in test_desc.keys()}
    return train_desc, test_desc, train_feats, test_feats

train_desc, test_desc, train_feats, test_feats = train_test_split(cap, all_features)

max_length = max_length(cap)


def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
      seq = tokenizer.texts_to_sequences([desc])[0]
      for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X1.append(photo)
        X2.append(in_seq)
        y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def create_data_sequences(desc, feats, tokenizer, max_length, vocab_len):
    data_images, data_sequences, data_labels = [], [], []

    for key, desc_list in desc.items():
        photo = feats[key][0]
        in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_len)
        data_images.extend(in_img)
        data_sequences.extend(in_seq)
        data_labels.extend(out_word)

    return [np.array(data_images), np.array(data_sequences)], np.array(data_labels)

# Define the Model

def define_model(vocab_size,max_caption_length):
  img_input=Input(shape=(4096,))
  drop=Dropout(0.3)(img_input)
  img_encoder=Dense(256, activation='relu')(drop)

  text_input = Input(shape=(max_caption_length,))
  text_embedding = Embedding(vocab_size, 256, mask_zero=True)(text_input)
  drop=Dropout(0.3)(text_embedding)
  text_encoder = LSTM(256)(drop)


  decoder1 = Add()([img_encoder, text_encoder])
  decoder2 = Dense(256, activation='relu')(decoder1)

  output = Dense(vocab_size, activation='softmax')(decoder2)
  model = Model(inputs=[img_input, text_input], outputs=output)

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  plot_model(model, to_file='model.png', show_shapes=True)
  return model

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
 while 1:
  for key, desc_list in descriptions.items():
    photo = photos[key][0]
    in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
    yield [in_img, in_seq], out_word

vocab_len=len(vocab)

train_data = create_sequences(tokenizer, max_length, train_desc, train_feats, vocab_len)

# Train the model

model = define_model(vocab_len, max_length)

epochs = 5
steps = len(train_desc)
for i in range(epochs):

 generator = data_generator(train_desc, train_feats, tokenizer, max_length, vocab_len)

 model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

 model.save('model_' + str(i) + '.keras')

# Testing and Evaluating Model

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = [],[]
    for key, desc_list in descriptions.items():
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())

    print('BLEU-1:', corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2:',corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3:',corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4:',corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    return actual,predicted

evaluate_model(model, test_desc, test_feats, tokenizer, max_length)