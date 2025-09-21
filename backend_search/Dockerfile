FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y git gcc g++ wget libssl-dev libncurses5-dev zlib1g-dev libffi-dev
WORKDIR /
RUN wget https://www.python.org/ftp/python/3.10.17/Python-3.10.17.tgz
RUN tar xzf Python-3.10.17.tgz
WORKDIR /Python-3.10.17
RUN ./configure --enable-optimizations
RUN make -j4
RUN make altinstall

RUN pip3.10 install onnxruntime
RUN pip3.10 install onnxruntime-gpu
RUN pip3.10 install insightface==0.7.3
RUN pip3.10 install pika
RUN pip3.10 install "numpy<2"
RUN pip install psycopg2
RUN pip install SQLAlchemy
WORKDIR /code
COPY . .
# ENV LD_LIBRARY_PATH /usr/local/cuda/lib64
RUN python3.10 download_model.py
# RUN cp /usr/local/cuda/lib64/libcublasLt.so.12 /usr/local/cuda/lib64/libcublasLt.so.11
# RUN cp /usr/local/cuda/lib64/libcufft.so.11 /usr/local/cuda/lib64/libcufft.so.10
# RUN cp /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.11.0
# RUN cp /usr/local/cuda/lib64/libcublas.so.12 /usr/local/cuda/lib64/libcublas.so.11
# RUN cp /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/x86_64-linux-gnu/libcudnn.so.8
CMD 'python3.10 worker_pytorch.py'
