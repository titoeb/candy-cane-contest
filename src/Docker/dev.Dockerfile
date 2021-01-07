FROM python:3.8-slim

RUN apt update && apt upgrade && apt-get install libgomp1

COPY requirements.txt ./ 

WORKDIR /usr/src

COPY . /usr/src 

RUN pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]
