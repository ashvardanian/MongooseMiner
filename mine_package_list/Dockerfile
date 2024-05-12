FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y gcc
RUN pip install pandas fastparquet uv dill --no-cache-dir
COPY . .

CMD python parse.py
