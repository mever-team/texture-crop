import os
import csv
from PIL import Image

def filter_large_images(folder, min_width=1024, min_height=1024):
    image_paths = []

    if folder in ["biggan", "crn", "cyclegan", "deepfake", "gaugan", "imle", "progan", 
                  "san", "seeingdark", "stargan", "stylegan", "stylegan2", "whichfaceisreal"]:
        folder_paths = [
            os.path.join("data/forensynths/test", folder, "1_fake"),
            os.path.join("data/forensynths/test", folder, "0_real")
        ]
    elif folder == "twigma":
        folder_paths = [
            "data/twigma/images",
            "data/openimagesdataset/train"
        ]
    else:
        folder_paths = [os.path.join("data/synthbuster", folder)]

    for folder_path in folder_paths:
        if "0_real" in folder_path or folder_path == "data/synthbuster/raise" or folder_path == "data/openimagesdataset/train":
            label = 0 
        else:
            label = 1  

        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if width >= min_width and height >= min_height:
                            rel_path = os.path.relpath(file_path, "data")
                            image_paths.append([rel_path, label])
                except Exception:
                    continue

    os.makedirs("datasets", exist_ok=True)
    with open(f"datasets/{folder}.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(image_paths)


if __name__ == "__main__":
    filter_large_images("twigma")

    for folder in os.listdir("data/forensynths/test"):
        if os.path.isdir(os.path.join("data/forensynths/test", folder)):
            filter_large_images(folder)

    for folder in os.listdir("data/synthbuster"):
        if os.path.isdir(os.path.join("data/synthbuster", folder)):
            filter_large_images(folder)
