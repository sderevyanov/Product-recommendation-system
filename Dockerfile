FROM python:3.9
# RUN apt-get update && apt-get install wget -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install libgomp1 -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/local/app_ml
RUN mkdir /usr/local/app_ml/logs #&& mkdir /usr/local/app_ml/models && mkdir /usr/local/app_ml/dataset
RUN /usr/local/bin/python3 -m pip install --upgrade pip
COPY . /usr/local/app_ml
RUN python3 -m pip install -r requirements.txt
EXPOSE 8015
CMD ["uvicorn", "src.skill_diplom_api.app:app", "--host", "0.0.0.0", "--port", "8015"]
#ADD . /usr/local/lightfm/
#WORKDIR /home/
#RUN cd lightfm && pip install -e .
