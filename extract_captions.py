import sys
from PIL import Image
sys.path.append(r'../MAGIC/image_captioning/language_model/')
sys.path.append(r'../MAGIC/image_captioning/clip/')

from simctg import SimCTG
from utlis import PlugAndPlayContrastiveDecodingOneStepFast
from clip import CLIP
import glob
import torch
import os
from tqdm import tqdm
import json
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # Load Language Model
        language_model_name = r'cambridgeltl/magic_mscoco'  # or r'/path/to/downloaded/cambridgeltl/magic_mscoco'
        self.sos_token, self.pad_token = r'<-start_of_text->', r'<-pad->'
        self.k, self.alpha, self.beta, self.decoding_len = 45, 0.1, 2.0, 16
        self.generation_model = SimCTG(language_model_name, self.sos_token, self.pad_token).cuda()

        model_name = r"openai/clip-vit-base-patch32"  # or r"/path/to/downloaded/openai/clip-vit-base-patch32"
        self.clip = CLIP(model_name).cuda()
        self.clip.cuda_available = True

        self.clip.eval()
        self.generation_model.eval()

        start_token = self.generation_model.tokenizer.tokenize(self.sos_token)
        self.start_token_id = self.generation_model.tokenizer.convert_tokens_to_ids(start_token)

    def forward(self, x):
        bs = len(x)
        input_ids = torch.LongTensor(self.start_token_id).unsqueeze(0).repeat(bs, 1).cuda()

        prefix_len = input_ids.size()[1]
        past_key_values, last_hidden_states, logits = None, None, None
        input_ids_for_class = input_ids.clone()

        visual_outputs = self.clip.model.vision_model(pixel_values=x)
        image_embeds = visual_outputs[1]
        image_embeds = self.clip.model.visual_projection(image_embeds)  # [1 x embed_dim]

        # the maximum supported length of generation for SimCTG is 256
        # to support longer generated length, you can re-train the SimCTG model with longer sequences
        decoding_len = self.decoding_len - prefix_len
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits, input_ids_for_class = \
                PlugAndPlayContrastiveDecodingOneStepFast(
                    self.generation_model,
                    input_ids,
                    prefix_len,
                    self.k,
                    self.alpha,
                    self.beta,
                    self.tokenizer,
                    image_embeds,
                    self.clip,
                    60,
                    past_key_values,
                    last_hidden_states,
                    logits,
                    first_step=step == 0,
                    input_ids_for_class=input_ids_for_class,
                )

        return [self.generation_model.parse_output_token_list(tokens) for tokens in input_ids_for_class]


class VideoDataset(Dataset):

    def __init__(self, paths):

        self.paths = paths
        self.processor = CLIPProcessor.from_pretrained(r"openai/clip-vit-base-patch32")

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']

        return image_path, pixel_values.squeeze(0)

    def __len__(self):
        return len(self.paths)

# # Load Language Model
# language_model_name = r'cambridgeltl/magic_mscoco'  # or r'/path/to/downloaded/cambridgeltl/magic_mscoco'
# sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
# k, alpha, beta, decoding_len = 45, 0.1, 2.0, 16
# num_sentences = 4
# generation_model = SimCTG(language_model_name, sos_token, pad_token)
# start_token = generation_model.tokenizer.tokenize(sos_token)
# start_token_id = generation_model.tokenizer.convert_tokens_to_ids(start_token)
# generation_model = torch.nn.DataParallel(generation_model)
# generation_model.eval()
# generation_model = generation_model.cuda()
#
# model_name = r"openai/clip-vit-base-patch32"  # or r"/path/to/downloaded/openai/clip-vit-base-patch32"
# clip = CLIP(model_name)
# clip.cuda_available = True
# clip = torch.nn.DataParallel(clip)
# clip.eval()
# clip = clip.cuda()

model = Model()
model = nn.DataParallel(model)
model = model.cuda()

folders = glob.glob(os.path.join("/mnt/hdd1", "Kinetics/Kinetics-700", "frames", "*"))
for folder in tqdm(folders):
    image_paths = sorted(glob.glob(os.path.join(folder, "images", "*")))
    dl = torch.utils.data.DataLoader(VideoDataset(image_paths),
                                     batch_size=128, num_workers=48, shuffle=False)
    this_json = dict()
    for (this_image_paths, pixel_values) in dl:
        texts = model(pixel_values.cuda())
        for i, image_path in enumerate(this_image_paths):
            keyname = os.path.basename(image_path).split(".")[0]
            this_json[keyname] = texts[i]

    with open(os.path.join(folder, "captions.json"), "w") as fp:
        json.dump(this_json, fp, indent=4, sort_keys=True)
