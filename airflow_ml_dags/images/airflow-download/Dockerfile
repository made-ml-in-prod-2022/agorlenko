FROM python:3.8-slim

RUN mkdir /build/

COPY requirements.txt /build/requirements.txt

RUN pip install -r /build/requirements.txt

COPY resources/data.csv /source_data.csv
COPY download.py /download.py

ENTRYPOINT ["python", "download.py"]
