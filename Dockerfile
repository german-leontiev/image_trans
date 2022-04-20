FROM python

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /usr/src
COPY requirements.txt .
RUN pip --no-cache install -r requirements.txt

COPY app.py .
COPY main.py .
COPY templates templates
COPY static static
CMD ["python3", "./main.py"]