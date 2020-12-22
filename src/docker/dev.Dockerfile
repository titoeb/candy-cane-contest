FROM python:3.8-slim

COPY environment/requirements.txt ./ 

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/src

CMD ["tail", "-f", "/dev/null"]
