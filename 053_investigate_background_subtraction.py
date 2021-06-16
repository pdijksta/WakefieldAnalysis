import pickle

import base64
from cam_server import PipelineClient

pipeline_client = PipelineClient("http://sf-daqsync-01:8889/")

bg = pipeline_client.get_latest_background("SARBD02-DSCR050")
image = pipeline_client.get_background_image_bytes(bg)
dtype = image["dtype"]
shape = image["shape"]
bytes = base64.b64decode(image["bytes"].encode())

with open('./bytes.pkl', 'wb') as f:
    pickle.dump(bytes, f)

