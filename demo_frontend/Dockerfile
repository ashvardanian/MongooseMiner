from python:3.10-slim

RUN pip install groq gradio infinity_emb[all] usearch datasets

WORKDIR /app
COPY . .

CMD bash entrypoint.sh