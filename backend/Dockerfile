FROM python:3.9-slim

RUN mkdir /backend
RUN mkdir /ml
RUN mkdir /data

RUN apt update && apt upgrade -y && apt install -y git

COPY ./backend/requirements.txt /backend/requirements.txt
RUN pip install -r backend/requirements.txt


COPY ./data /data
COPY ./backend /backend
COPY ./ml /ml

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
