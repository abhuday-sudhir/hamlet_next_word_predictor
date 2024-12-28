import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model=load_model("LSTM_NEXTWORD_PREDICTOR.h5")

with open('tokenizer.pickle','rb') as file:
    tokenizer=pickle.load(file)


def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if(len(token_list))>=max_sequence_len:
        token_list=token_list[-(max_sequence_len):]
    input_sequence=pad_sequences([token_list],maxlen=max_sequence_len)
    prediction=model.predict(input_sequence)
    predicted_word_index=np.argmax(prediction,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

input=st.text_input("Enter the sentence for next word prediction")

if st.button("Predict next word"):
    predicted_word=predict_next_word(model,tokenizer,input,model.input_shape[1])
    st.write(f"The next word is {predicted_word}")