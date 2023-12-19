  # Image captioning
Image captioning is a fascinating and challenging task in the field of computer vision and natural language processing. The goal of an image captioning project is to develop a system that can generate human-like textual descriptions for images. Essentially, it combines computer vision techniques to understand the content of an image and natural language processing to express that understanding in the form of a coherent and descriptive sentence.



Dataset and its structure

1. Commonly known datasets that can be used for trianing purpose:
    1. Flickr8K
    2. Flick30K
    3. Fick100K
    4. MSCOCO
2. Each dataset may have there own structure of dataset. For Flickr_8K dataset, all the images of training, validation and test set are in one folder. It contains 3 different files i.e Flickr_8k.trainImages.txt, Flickr_8k.testImages.txt , Flickr_8k.devImages.txt  corresponding to each type of dataset i.e train, test and validation set, each file having file_name of images conatined in each dataset. 
3. For example, in Flick8k, Flickr_8k.trainImages.txt file contains file_ids of images in training set. Name of image file is its image id.
4. All the images are in same folder. So to parse images of training dataset, first read trianImages.txt file, read line by line image id and load corresponding image from image dataset folder.
5. Each image is given 5 different captions by 5 different humans. This is because an image can be described in multiple ways.


One measure that can be used to evaluate the skill of the model are BLEU scores. For reference, below are some ball-park BLEU scores for skillful models when evaluated on the test dataset (taken from the 2017 paper “Where to put the Image in an Image Caption Generator“ [https://arxiv.org/abs/1703.09137]):

    BLEU-1: 0.401 to 0.578.
    BLEU-2: 0.176 to 0.390.
    BLEU-3: 0.099 to 0.260.
    BLEU-4: 0.059 to 0.170.

Model

Objective: Image captioning, generating textual descriptions for images.
Preprocessing:
  Image features extracted using VGG16.
  Text tokenized and embedded (size 40).
Model Architecture:
  Combines CNN (VGG16) and LSTM for image and text processing.
  Includes dropout layers for regularization.
  Output is a probability distribution over the vocabulary for generating captions.
Training:
  Number of epochs: 5.
BLEU Scores:
  BLEU-1: 0.4851
  BLEU-2: 0.3028
  BLEU-3: 0.2084
  BLEU-4: 0.1013

![model](https://github.com/tonystark2032/ML-Codes/assets/116151399/5ef93135-dc55-4ec5-bfbe-2709302fffff)

