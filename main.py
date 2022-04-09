from matplotlib.font_manager import json_load
from yt_dlp import YoutubeDL as yt
import re
import json

parent_id = "j64oZLF443g"


def extract_info(id):

    url = "https://www.youtube.com/watch?v="
    info = yt({}).extract_info(url+id,download = False)
    temp = {
        "id": info["id"],
        "title": info["title"],
        "thumbnail": info["thumbnail"],
        "description": info["description"]
    }
    info = temp

    info["children"] = re.findall(r"youtu\.be/(.{11})",temp["description"])

    return info


def record_info(id):
    if id not in mega_dict: 
        info = extract_info(id)
        mega_dict[id] = info
        with open("nodes.json", "w") as f:
            json.dump(mega_dict, f,indent=4)
        for child_id in info["children"]:
            record_info(child_id)

if __name__ == "__main__":
    with open("nodes.json", "r") as f:
        mega_dict = json.load(f)
    record_info(parent_id)





