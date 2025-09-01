import pika
import cv2
import os
import numpy as np
from io import BytesIO
import time
import json
from insightface.app import FaceAnalysis


path = '/mnt/data/'
device = 'cpu'
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU


def consine_similrity(emb1, emb2,):
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


def process(file_bytes):
    try:
        img_np = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        faces = app.get(img)
        print(len(faces))
        outputs = {'url': [], 'confidence': []}
        if len(faces) == 0:
            return outputs

        query_embed = faces[0].embedding
        for i, file in enumerate(os.listdir(os.path.join(path, 'features'))):
            print(i)
            db_embeds = np.load(os.path.join(path, 'features', file))
            max_sims = 0
            for db_embed in db_embeds:
                sim = consine_similrity(query_embed, db_embed)
                if sim > max_sims:
                    max_sims = sim
            if max_sims > 0.6:
                outputs['url'].append(os.path.join('', file[:-4]))
                outputs['confidence'].append(max_sims.tolist())

        return outputs
    except Exception as e:
        return {'error': str(e), 'success': False}


def on_request(ch, method, props, body):
    response = process(body)
    if isinstance(response, dict):
        response = json.dumps(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=response)
    ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    url = os.environ.get('CLOUDAMQP_URL', 'amqp://guest:guest@search-mq/%2f')  # Taz check!
    time.sleep(5)
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)

    channel = connection.channel()
    channel.queue_declare(queue='image_queue')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='image_queue', on_message_callback=on_request)

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
