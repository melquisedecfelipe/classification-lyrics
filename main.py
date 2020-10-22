import joblib
import numpy as np

from fastapi import FastAPI, Body
from pydantic import BaseModel

app = FastAPI()

class Payload(BaseModel):
  lyrics: str

@app.get("/")
def home():
  return {'message': 'See docs: https://classification-unip.herokuapp.com/docs'}

@app.post('/api/lyrics/v1')
def classification_lyrics(payload: Payload):
  model = joblib.load('classificationLyrics.unip')

  wordsCounter = len(payload.lyrics.split())
  minimumWords = 50

  if (wordsCounter < minimumWords):
    return { 'wordsCounter': wordsCounter, 'minimumWords': minimumWords, 'message': 'Quantidade mínima de palavras não atingida, por favor, considere pegar uma parte da letra maior.' }
  
  band = model.predict([payload.lyrics])
  probability = model.predict_proba([payload.lyrics])
  index = np.argmax(probability)

  data = { 'band': band[0], 'probability': probability[0][index], 'lyrics': payload.lyrics }

  return data