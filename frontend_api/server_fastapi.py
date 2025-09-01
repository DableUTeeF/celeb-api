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


app = FastAPI()
app.mount("/static", StaticFiles(directory="/mnt/static"), name="static")


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


def process_img(cvimg1, cvimg2, org_filename=''):
    client = RPCClient()
    result = client.call(cvimg, 'image_queue')
    return result


def process_video(cvimg, org_filename=''):
    client = RPCClient()
    result = client.call(cvimg, 'video_queue')
    return result


@app.post('/image')
def _file_upload(image1: UploadFile, image2: UploadFile, request: Request):
    results = process_img(image1.file.read(), image2.file.read())
    if isinstance(results, bytes):
        results = json.loads(results.decode())

        return JSONResponse(content=results)
    elif 'Image is too big.' in results['error']:
        return JSONResponse(content={'error': results['error']}, status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    else:
        return JSONResponse(content={'error': results['error']}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post('/video')
def _file_upload(file: UploadFile, request: Request):
    results = process_img(file.file.read())
    if isinstance(results, bytes):
        results = json.loads(results.decode())

        return JSONResponse(content=results)
    elif 'Image is too big.' in results['error']:
        return JSONResponse(content={'error': results['error']}, status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    else:
        return JSONResponse(content={'error': results['error']}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


if __name__ == "__main__":
    uvicorn.run("server_fastapi:app", host="0.0.0.0", port=8000, log_level="debug")
