from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
# app.prepare(ctx_id=0)  # ctx_id=-1 for CPU, 0 for GPU

