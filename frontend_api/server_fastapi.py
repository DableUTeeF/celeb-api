from fastapi import FastAPI, File, UploadFile, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
import pika
import os
import uuid
from PIL import Image
import json
import base64
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine


app = FastAPI()
app.mount("/static", StaticFiles(directory="/mnt/static"), name="static")

engine = create_engine("postgresql+psycopg2://postgres:postgres@db:5432/postgres", echo=True)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "user_account"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]
    addresses: Mapped[List["Address"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"


class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str]
    user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"))
    user: Mapped["User"] = relationship(back_populates="addresses")

    def __repr__(self) -> str:
        return f"Address(id={self.id!r}, email_address={self.email_address!r})"


class RPCClient(object):
    def __init__(self):
        url = os.environ.get('CLOUDAMQP_URL', 'amqp://guest:guest@search-mq/%2f')
        params = pika.URLParameters(url)
        self.connection = pika.BlockingConnection(params)

        self.channel = self.connection.channel()

        result = self.channel.queue_declare('', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, body, routing_key):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key=routing_key,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=body)
        while self.response is None:
            self.connection.process_data_events()
        return self.response

    def callasync(self, body, routing_key):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key=routing_key,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=body
        )


@app.post('/images')
def images_similarity(image1: UploadFile, image2: UploadFile, request: Request):
    client = RPCClient()
    img1_np = np.frombuffer(image1.file.read(), np.uint8)
    img2_np = np.frombuffer(image2.file.read(), np.uint8)
    img1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)
    _, buffer1 = cv2.imencode('.jpg', img1)
    _, buffer2 = cv2.imencode('.jpg', img2)
    results = client.call(
        json.dumps({'data': {'image1': base64.b64encode(buffer1).decode(), 'image2': base64.b64encode(buffer2).decode()}, 'type': 'images_sim'}),
        'image_queue'
    )

    if isinstance(results, bytes):
        results = json.loads(results.decode())

        return JSONResponse(content=results)
    elif 'Image is too big.' in results['error']:
        return JSONResponse(content={'error': results['error']}, status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    else:
        return JSONResponse(content={'error': results['error']}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post('/image-feature')
def image_feature(image: UploadFile, request: Request):
    client = RPCClient()
    img1_np = np.frombuffer(image.file.read(), np.uint8)
    img1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
    _, buffer1 = cv2.imencode('.jpg', img1)
    results = client.call(
        json.dumps({'data': {'image': base64.b64encode(buffer1).decode()}, 'type': 'images_features'}),
        'image_queue'
    )
    client.connection.close()
    if isinstance(results, bytes):
        results = json.loads(results.decode())

        return JSONResponse(content=results)
    elif 'Image is too big.' in results['error']:
        return JSONResponse(content={'error': results['error']}, status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    else:
        return JSONResponse(content={'error': results['error']}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post('/video')
def video_search(image: UploadFile, video: UploadFile, skip_interval: int, request: Request):
    client = RPCClient()
    contents = video.file.read()
    temp = NamedTemporaryFile(delete=False)
    with temp as f:
        f.write(contents)
    img1_np = np.frombuffer(image.file.read(), np.uint8)
    img1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
    _, buffer1 = cv2.imencode('.jpg', img1)
    client.callasync(
        json.dumps({'data': {'image': base64.b64encode(buffer1).decode(), 'video': temp.name, 'skip_interval': skip_interval}, 'type': 'video'}),
        'image_queue'
    )
    # client.connection.close()

    return JSONResponse(content={'file_id': temp.name[5:]+'.json'})


if __name__ == "__main__":
    uvicorn.run("server_fastapi:app", host="0.0.0.0", port=8000, log_level="debug")
