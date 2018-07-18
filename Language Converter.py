"""
This is a RECURRENT NEURAL NETWORK for Language Translation (english to Hindi) using Keras.
"""
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint

input_text = []
target_text = []
latent_dim = 256
samples = 10000
num_words = 5000

#Data Loading and preprocessing
path1 = r'eng10000.txt'
path2 = r'hi10000.txt'
with open(path1, 'r') as f:
    input_text = f.read().split('\n')

with open(path2, 'r', encoding = 'utf-8') as f:
    target_text = f.read().split('\n')

#Inserting <start_of_sequence> and <end of sequence>
for i,txt in enumerate(target_text):
    target_text[i] = 'sos '+txt+' eos'

tokenizer1 = Tokenizer(num_words = num_words)
tokenizer1.fit_on_texts(input_text)
input_text = tokenizer1.texts_to_sequences(input_text)

tokenizer2 = Tokenizer(num_words = num_words)
tokenizer2.fit_on_texts(target_text)
target_text = tokenizer2.texts_to_sequences(target_text)

enc_timesteps = max([len(i) for i in input_text])
dec_timesteps = max([len(i) for i in target_text])

enc_input = np.zeros((len(input_text), enc_timesteps, num_words), dtype = 'uint8')
dec_input = np.zeros((len(target_text), dec_timesteps, num_words), dtype = 'uint8')
dec_target = np.zeros(dec_input.shape, dtype = 'uint8')

for i in range(enc_input.shape[0]):
    for j,k in enumerate(input_text[i]):
        enc_input[i,j,k] = 1

for i in range(dec_input.shape[0]):
    for j,k in enumerate(target_text[i]):
        dec_input[i,j,k] = 1
        if j>0:
            dec_target[i,j-1,k] = 1

#Defining Architecture of the network
#Defining ENCODER [It takes encoder input and return the encoded information which is internal 
#state of the LSTM layer]
enc_inp = Input(shape = (None, enc_input.shape[-1]))
enc_lstm = LSTM(latent_dim, return_state = True)
_, state_h, state_c = enc_lstm(enc_inp)
states = [state_h, state_c]
enc_model = Model(enc_inp, states)

#Defining DECODER
dec_inp = Input(shape = (None, dec_input.shape[-1]), name = 'Decoder_input')
dec_lstm = LSTM(latent_dim, return_state = True, return_sequences = True)
dec_out, _, _ = dec_lstm(dec_inp, initial_state = states)
dec_dense = Dense(num_words, activation = 'softmax')
dec_out = dec_dense(dec_out)
train_model = Model([enc_inp, dec_inp], dec_out)
#Since train_model takes dec_inp as one of the inputs in order to predict the output
#We need to define another model for prediction

#Let's define a method to predict the output for an input sequence which is a string
inp_state_h = Input(shape = (latent_dim,), name = 'state_h')
inp_state_c = Input(shape = (latent_dim,), name = 'state_c')
inp_states = [inp_state_h, inp_state_c]
dec_out, dec_state_h, dec_state_c = dec_lstm(
    dec_inp, initial_state=inp_states)
dec_states = [dec_state_h, dec_state_c]
dec_out = dec_dense(dec_out)
predict_model = Model([dec_inp]+inp_states, [dec_out]+dec_states)

def predict(inp_str):
    print("you entered -> "+inp_str)
    text_inputt = tokenizer1.texts_to_sequences([inp_str])
    enc_inpp = np.zeros((1, enc_timesteps, num_words))
    for i,j in enumerate(text_inputt[-1]):
        enc_inpp[0,i,j] = 1
    
    statess = enc_model.predict(enc_inpp)
    res = ''
    wi = tokenizer2.word_index
    rev_wi = {i:ch for ch,i in wi.items()}
    dec_inpp = np.zeros((1, 1, num_words))
    dec_inpp[0,0,wi['sos']] = 1
    last = wi['eos']
    maxlen = 50
    while True:
        dec_outt, h, c = predict_model.predict([dec_inpp]+statess)
        index = np.argmax(np.squeeze(dec_outt))
        if index == last or len(res.split())>=maxlen:
            break
        res = res + rev_wi[index]+' '
        statess = [h, c]
        dec_inpp = np.zeros((1, 1, num_words))
        dec_inpp[0,0,index] = 1
    
    return res

    
#Lets train our model
clbk = ModelCheckpoint(filepath = 'eng_hindi_best_model.h5', monitor = 'val_loss', save_best_only=True)
train_model.compile('adam', 'categorical_crossentropy', metrics = ['accuracy'])
history = train_model.fit([enc_input, dec_input], dec_target, epochs = 1, batch_size = 64,
                validation_split = 0.1, callbacks = [clbk])

train_model.save('eng_hindi_model.h5')

inp = 'y'
while(inp == 'y'):
    inp_str = input('Enter text\n')
    res = predict(inp_str)
    print(res)
    inp = input('Wanna try again?\n [y/n]')
