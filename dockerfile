FROM tiangolo/uwsgi-nginx-flask:python3.8  
COPY ./requirements.txt /app/requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY . /app
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# RUN python3 -m pip install -r requirements.txt
