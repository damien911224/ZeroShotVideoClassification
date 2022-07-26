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

# Load Language Model
language_model_name = r'cambridgeltl/magic_mscoco'  # or r'/path/to/downloaded/cambridgeltl/magic_mscoco'
sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
k, alpha, beta, decoding_len = 45, 0.1, 2.0, 16
num_sentences = 4
generation_model = SimCTG(language_model_name, sos_token, pad_token).cuda()
generation_model.eval()

model_name = r"openai/clip-vit-base-patch32"  # or r"/path/to/downloaded/openai/clip-vit-base-patch32"
clip = CLIP(model_name).cuda()
clip.eval()

frame_paths = glob.glob(os.path.join("/mnt/hdd1", "Kinetics/Kinetics-700", "frames", "*", "images", "*"))


with torch.no_grad():
    start_token = generation_model.tokenizer.tokenize(sos_token)
    start_token_id = generation_model.tokenizer.convert_tokens_to_ids(start_token)
    input_ids = torch.LongTensor(start_token_id).unsqueeze(0)

    word_feats = list()
    word_samples = list()
    x = buffer.squeeze(0)
    image = ((x.permute(1, 2, 3, 0).numpy() * 2 + 1) * 255.0).astype(np.uint8)
    image_indices = np.linspace(0, s[1] - 1, num_sentences, dtype=np.int32)
    for image_index in image_indices:
        image_instance = Image.fromarray(image[image_index])
        w_feats, tokens = \
            generation_model.magic_search(input_ids, k, alpha, decoding_len,
                                               beta, image_instance, clip, 60)
        word_feats.append(w_feats.squeeze(0))

        text = generation_model.tokenizer.decode(tokens.squeeze(0)).strip()
        text = ' '.join(text.split()).strip()
        word_samples.append(text)
    word_feats = torch.cat(word_feats, dim=0)