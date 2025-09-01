FROM python:3.9.22
WORKDIR /app
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install python-multipart
RUN pip install pika
RUN pip install pillow
COPY . .
RUN mkdir static
CMD 'python server_fastapi.py'
