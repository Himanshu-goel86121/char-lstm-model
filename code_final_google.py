import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
data=pd.read_csv("E://en_train.csv",encoding="utf-8")
data = data[pd.notnull(data['before'])]
data = data[pd.notnull(data['after'])]
data['before']=data['before'].astype('str')
input = data['before'].values.tolist()
tok=Tokenizer(char_level=True,filters='')
tok.fit_on_texts(input)
input=tok.texts_to_sequences(input)
sequence_input=pad_sequences(input,maxlen=32)
output = data['class'].values.tolist()
out_list = set(output)
out_map = {}
count=0
for each in out_list:
    out_map[each]=count
    count+=1

np.save('out_map.npy',out_map)
import pickle
pickle.dump(tok,open('google_model_tokenizer.pkl','wb'))
outty=[]
for each in output:
    outty.append(out_map[each])

print(out_map.keys())

word_index=tok.word_index
from gensim.models import Word2Vec
model = Word2Vec.load('word2vec_char_model.model')
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    try:
        embedding_vector = embeddings_index.get(word)
    except:
        embedding_vector = None
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.layers import LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping
x=Input(shape=(32,))
y=Embedding(len(word_index) + 1,100,weights=[embedding_matrix],input_length=32)(x)
y=LSTM(128, return_sequences=True)(y)
y=LSTM(128)(y)
y=Dense(16,activation='softmax')(y)
model=Model(inputs=x,outputs=y)
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
clbck=[EarlyStopping(patience=1)]
model.fit(sequence_input,outty,batch_size=1000,epochs=3,validation_split=0.1,callbacks=clbck)
model.save('google_model.model')