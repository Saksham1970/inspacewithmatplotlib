import json
from PIL import Image
import requests
from io import BytesIO

with open("nodes.json", "r") as f:
    mega_dict = json.load(f)

for node in mega_dict:
    mega_dict[node]["thumbnail"] = Image.open(BytesIO(requests.get(mega_dict[node]["thumbnail"]).content))

import pickle
with open("nodes.nvm","wb") as f:
    pickle.dump(mega_dict,f)
