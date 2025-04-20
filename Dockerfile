FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "dataflow.py", "--server.port=8501", "--server.address=0.0.0.0"]