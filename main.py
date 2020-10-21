import joblib
import numpy as np

from fastapi import FastAPI, Body

app = FastAPI()

@app.get("/")
def home():
  return {'message': 'See docs: https://classification-unip.herokuapp.com/docs'}

@app.post('/api/lyrics/v1')
def classification_lyrics(lyrics: str):
  model = joblib.load('model/classificationLyrics.unip')
  
  band = model.predict([lyrics])
  probability = model.predict_proba([lyrics])
  index = np.argmax(probability)

  data = { 'band': band[0], 'probability': probability[0][index], 'lyrics': lyrics }

  return data