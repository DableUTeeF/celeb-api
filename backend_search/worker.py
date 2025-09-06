import pika
import cv2
import os
import numpy as np
from io import BytesIO
import time
import json
from insightface.app import FaceAnalysis
import base64


path = '/mnt/data/'
device = 'cpu'
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=0)  # ctx_id=-1 for CPU, 0 for GPU


def consine_similrity(emb1, emb2,):
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


def images_similarity(body):
    try:
        img1_np = np.frombuffer(base64.b64decode(body['data']['image1']), np.uint8)
        img2_np = np.frombuffer(base64.b64decode(body['data']['image2']), np.uint8)
        img1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)
        face1 = app.get(img1)
        face2 = app.get(img2)
        outputs = {'similarity': -1}
        if len(face1) + len(face2) == 0:
            return outputs

        face1_embed = face1[0].embedding
        face2_embed = face2[0].embedding
        sim = consine_similrity(face1_embed, face2_embed).max()
        outputs['similarity'] = sim.tolist()

        return outputs
    except Exception as e:
        return {'error': str(e), 'success': False}


def images_features(body):
    try:
        img_np = np.frombuffer(base64.b64decode(body['data']['image']), np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        faces = app.get(img)
        outputs = []
        if len(faces) == 0:
            return outputs
        for face in faces:
            face_embed = face.embedding
            face_bbox = face.bbox
            face_gender = face.gender
            face_age = face.age
            face_landmark = face.landmark_2d_106
            outputs.append(
                {
                    'bbox': face_bbox.tolist(),
                    'features': face_embed.tolist(),
                    'landmark': face_landmark.tolist(),
                    'gender': ['female', 'male'][face_gender],
                    'age': face_age,
                }
            )

        return {'results': outputs}
    except Exception as e:
        return {'error': str(e), 'success': False}


def video_process(body):
    try:
        img_np = np.frombuffer(base64.b64decode(body['data']['image']), np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        query_faces = app.get(img)
        outputs = {'results': [], 'summary': {'appearances': []}}
        if len(query_faces) == 0:
            return outputs
        query_face_embed = query_faces[0].embedding

        vids = cv2.VideoCapture(body['data']['video'])
        frame_idx = -1
        while vids.isOpened():
            frame_idx += 1
            _, frame = vids.read()
            if frame_idx % body['data']['skip_interval'] > 0:
                continue
            if frame is None:
                break
            frame_faces = app.get(frame)
            frame_result = {'frame_index': frame_idx, 'faces': []}
            appeared = False
            if len(frame_faces) > 0:
                for face in frame_faces:
                    face_embed = face.embedding
                    face_bbox = face.bbox
                    face_gender = face.gender
                    face_age = face.age
                    face_landmark = face.landmark_2d_106
                    sim = consine_similrity(query_face_embed, face_embed).max()
                    frame_result['faces'].append(
                        {
                            'bbox': face_bbox.tolist(),
                            'similarity': sim.tolist(),
                            'landmark': face_landmark.tolist(),
                            'gender': ['female', 'male'][face_gender],
                            'age': face_age,
                        }
                    )
                    if sim > 0.6:
                        appeared = True
            outputs['results'].append(frame_result)
            if appeared:
                outputs['summary']['appearances'].append(frame_idx)

        return outputs
    except Exception as e:
        return {'error': str(e), 'success': False}


def on_request(ch, method, props, body):
    body = json.loads(body)
    if body['type'] == 'images_sim':
        response = images_similarity(body)
    elif body['type'] == 'images_features':
        response = images_features(body)
    elif body['type'] == 'video':
        response = video_process(body)
        with open(f'/mnt/static/{body["data"]["video"][5:]}.json', 'w') as wr:
            json.dump(response, wr)
        return
    else:
        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id=props.correlation_id),
                         body='invalid type')
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return
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
