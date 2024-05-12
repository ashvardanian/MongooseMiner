FROM python:3.10-slim

WORKDIR /app
RUN pip install pandas fastparquet uv dill --no-cache-dir
COPY . .

CMD python parse.py
