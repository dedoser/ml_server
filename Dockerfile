FROM python:3-slim-buster

WORKDIR /server

RUN mkdir data

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/server/

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]