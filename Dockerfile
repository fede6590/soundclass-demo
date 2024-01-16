FROM python:3.11-slim-bookworm
 
RUN useradd -m dev
 
WORKDIR /app
 
RUN apt-get update && apt-get install -y \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER dev

COPY . /app/
 
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt
 
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
# CMD ["streamlit", "run", "--server.address=0.0.0.0", "app.py"]
