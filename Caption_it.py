#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from nltk.corpus import stopwords
import string
import json
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


# In[3]:


#after training the model, the model_weights we got is being loaded
model=load_model("./model_weights/model_9.h5")

model._make_predict_function()


# In[4]:


# when the new image will be given first we need to calcuate encoding for the image. For that purpose we will 
# use ResNet50
model_temp=ResNet50(weights="imagenet",input_shape=(224,224,3))


# In[5]:


model_resnet = Model(model_temp.input, model_temp.layers[-2].output)
model_resnet._make_predict_function() 
# In[6]:


def preprocess_image(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img, axis=0) 
    img=preprocess_input(img)
    return img


# In[12]:


def encode_image(img):
    img=preprocess_image(img)
    feature_vector=model_resnet.predict(img)
    feature_vector=feature_vector.reshape((1,feature_vector.shape[1]))
    return feature_vector


# In[13]:





# In[14]:




# In[26]:


with open("./storage/word_to_idx.pkl","rb") as w2i:
    word_to_idx=pickle.load(w2i)


# In[23]:


with open("./storage/idx_to_word.pkl","rb") as i2w:
    idx_to_word=pickle.load(i2w)


# In[27]:


def predict_caption(photo):
    in_text = "startseq"
    max_len=35
    for i in range(max_len):
        sequence=[word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence=pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred=model.predict([photo,sequence])
        ypred=ypred.argmax() # Word With max Possibility (Greedy Sampling)
        word=idx_to_word[ypred] # GEt actual Word
        in_text+=' '+word # append to in_text
        
        if word=='endseq': # we can break as end seq is found
            break
        
        
    final_caption=in_text.split()
    final_caption=final_caption[1:-1]
    final_caption=' '.join(final_caption)
    
    return final_caption


# In[28]:

def caption_this_image(image):
	enc=encode_image(image)
	caption=predict_caption(enc)
	return caption

# In[ ]:





# In[ ]:





# In[ ]:




