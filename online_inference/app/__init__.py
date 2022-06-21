import asyncio
import logging
import os
import pickle
import sys
import tempfile
from typing import Optional, List

import gdown
import uvicorn
from fastapi import FastAPI
from sklearn.pipeline import Pipeline

from .prediction import make_predict
from .validation import DiseasePredictResponse, HeartDiseasesModel


STARTUP_DELAY = 20
LIFETIME = 120


logger = logging.getLogger(__name__)


app = FastAPI()
model: Optional[Pipeline] = None


async def kill_app():
    await asyncio.sleep(LIFETIME)
    sys.exit(1)


@app.on_event('startup')
async def load_model():
    global model
    model_url = os.getenv('MODEL_URL')
    if model_url is None:
        err = f'MODEL_URL is None'
        logger.error(err)
        raise RuntimeError(err)
    with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp_file:
        logger.info('start downloading model')
        gdown.download(model_url, output=tmp_file.name)
        logger.info('finish downloading model')
        with open(tmp_file.name, mode='rb') as model_file:
            model = pickle.load(model_file)
    await asyncio.sleep(STARTUP_DELAY)
    asyncio.create_task(kill_app())


@app.get('/health')
async def get_status():
    return model is not None


@app.post("/predict", response_model=List[DiseasePredictResponse])
async def predict(data: HeartDiseasesModel):
    return make_predict(data.dict(), model)


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=os.getenv('PORT', 1234))
