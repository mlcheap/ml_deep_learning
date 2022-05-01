FROM python:3.8.10
WORKDIR /app
COPY . /app
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install -r requirements.txt
CMD ["python3 -m flask run --host=0.0.0.0 --port=5000" ]
