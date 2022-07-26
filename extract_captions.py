import sys
from PIL import Image
sys.path.append(r'../MAGIC/image_captioning/language_model/')
sys.path.append(r'../MAGIC/image_captioning/clip/')

from simctg import SimCTG
from clip import CLIP
import glob
import torch
import os
from tqdm import tqdm
import json

# Load Language Model
language_model_name = r'cambridgeltl/magic_mscoco'  # or r'/path/to/downloaded/cambridgeltl/magic_mscoco'
sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
k, alpha, beta, decoding_len = 45, 0.1, 2.0, 16
num_sentences = 4
generation_model = SimCTG(language_model_name, sos_token, pad_token)
generation_model = nn.DataParallel(generation_model)
generation_model.eval()
generation_model = generation_model.cuda()

model_name = r"openai/clip-vit-base-patch32"  # or r"/path/to/downloaded/openai/clip-vit-base-patch32"
clip = CLIP(model_name)
clip.cuda_available = True
clip = nn.DataParallel(clip)
clip.eval()
clip = clip.cuda()

start_token = generation_model.tokenizer.tokenize(sos_token)
start_token_id = generation_model.tokenizer.convert_tokens_to_ids(start_token)

folders = glob.glob(os.path.join("/mnt/hdd1", "Kinetics/Kinetics-700", "frames", "*"))
with torch.no_grad():
    for folder in tqdm(folders):
        image_paths = glob.glob(os.path.join(folder, "images", "*"))
        this_json = dict()
        # for image_path in image_paths:
        #     keyname = os.path.basename(image_path).split(".")[0]
        #     image_instance = Image.open(image_path)
        #     text = generation_model.magic_search(input_ids, k, alpha, decoding_len, beta, image_instance, clip, 60)
        #     this_json[keyname] = text

        input_ids = torch.LongTensor(start_token_id).unsqueeze(0).repeat(len(image_paths), 1).cuda()
        image_instance = [Image.open(image_path) for image_path in image_paths]
        texts = generation_model.magic_search(input_ids, k, alpha, decoding_len, beta, image_instance, clip, 60)
        for i, image_path in enumerate(image_paths):
            keyname = os.path.basename(image_path).split(".")[0]
            this_json[keyname] = texts[i]

        with open(os.path.join(folder, "captions.json"), "w") as fp:
            json.dump(this_json, fp, indent=4, sort_keys=True)
