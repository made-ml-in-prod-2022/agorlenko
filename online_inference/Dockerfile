FROM python:3.9-slim

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app

ENV MODEL_URL="https://drive.google.com/uc?export=download&id=1OMi2-Z3pUp4ObTQC2l-2hh0blKYbWwvl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
