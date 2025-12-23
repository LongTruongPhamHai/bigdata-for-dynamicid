import torch
import os
from pipeline import DynamicIDStableDiffusionPipeline
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
import torch
import numpy as np

face_analyzer = FaceAnalysis(
    name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_analyzer.prepare(ctx_id=0, det_size=(512, 512))


device = "cuda"

# ------------- FIX -------------
# base_model_path = "./models/Realistic_Vision_V6.0_B1_noVAE"
base_model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
# -------------------------------

# ------------- FIX -------------
# SAA_path = "./models/SAA.bin"
SAA_path = "/kaggle/working/bigdata-for-dynamicid/models/SAA.bin"
# -------------------------------

pipe = DynamicIDStableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
).to(device)

# ------------- FIX -------------
# pipe.load_DynamicID(SAA_path)
pipe.load_DynamicID(base_model_path, SAA_path, IMR_path=None)

# -------------------------------

# ------------- FIX -------------
# root_path = "./dataset/base_image_dataset"
root_path = "/kaggle/input/dataset-for-dynamicid/dataset/base_image_dataset"
# -------------------------------

num = len(os.listdir(root_path))
for i in range(num):
    person_path = os.path.join(root_path, str(i))
    image_path = sorted(os.listdir(person_path))

    # ------------- FIX -------------
    # save_path = person_path.replace("base_image_dataset", "cache")
    save_path = person_path.replace(
        "/kaggle/input/dataset-for-dynamicid/dataset/base_image_dataset",
        "/kaggle/working/cache",
    )
    # -------------------------------

    os.makedirs(save_path, exist_ok=True)

    for path in image_path:
        if path.endswith(".txt") or path.endswith("landmark.png"):
            continue
        select_image = load_image(os.path.join(person_path, path))

        # ------------- FIX -------------
        # face_embed = pipe.cal_face_embed(select_image)
        face_embed = cal_face_embed(select_image)
        # -------------------------------

        if face_embed is None:
            print(f"Error in {str(i)}_{path}")
        else:
            torch.save(face_embed, os.path.join(save_path, path.split(".")[0] + ".pt"))


# ------------- FIX -------------
def cal_face_embed(image_pil):
    img = np.array(image_pil)
    faces = face_analyzer.get(img)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
    return torch.from_numpy(face.embedding).float()


# -------------------------------
