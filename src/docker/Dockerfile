FROM python:3.8-slim

COPY requirements.txt ./ 

WORKDIR /usr/src

COPY ./ /usr/src 

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python "]
