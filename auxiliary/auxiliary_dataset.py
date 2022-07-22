import os, numpy as np
from time import time

import cv2, torch
from torch.utils.data import Dataset
from auxiliary.transforms import get_transform
from scipy.spatial.distance import cdist
import json
import glob

from gensim.models import KeyedVectors as Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm
import random

import re
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

def get_ucf101():
    # folder = '/mnt/hdd1/UCF101/videos'
    folder = '/mnt/hdd1/UCF101/frames'
    fnames, labels = [], []
    paths = sorted(glob.glob(os.path.join(folder, "*")))
    for path in paths:
        # for fname in os.listdir(os.path.join(str(folder), label)):
        fname = os.path.basename(path)
        label = fname.split("_")[1]

        # words = [[label[0]]]
        #
        # if label not in ["TaiChi", "YoYo"]:
        #     for c in label[1:]:
        #         if words[-1][-1].islower() and c.isupper():
        #             words.append(list(c))
        #         else:
        #             words[-1].append(c)
        #
        #     label = " ".join([''.join(word) for word in words])

        fnames.append(path)
        labels.append(label)

    classes = np.unique(labels)
    return fnames, labels, classes


def get_hmdb():
    root_folder = os.path.join("/mnt/hdd1/HMDB51")
    with open(os.path.join(root_folder, "hmdb51.json"), "r") as fp:
        gt_json = json.load(fp)
    class_map = dict()
    with open(os.path.join(root_folder, "hmdb51_classes.txt"), "r") as fp:
        while True:
            line = fp.readline().rstrip()
            if not line:
                break
            name, idx = line.split(" ")
            class_map[int(idx)] = name

    fnames, labels = [], []
    # paths = sorted(glob.glob(os.path.join(root_folder, "videos", "*")))
    paths = sorted(glob.glob(os.path.join(root_folder, "frames", "*")))
    for fname in paths:
        fnames.append(fname)
        label = class_map[gt_json["database"][os.path.basename(fname).split(".")[0]]["annotations"]]
        labels.append(label.replace('_', ' '))

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes


'''
This function is ad-hoc to my personal format of kinetics.
You need to adjust it to your data format.
'''
def get_kinetics(dataset='k700'):
    sourcepath = '/mnt/hdd1/Kinetics'
    # n_classes = '700' if '700' in dataset else '400'
    n_classes = '700'
    # with open(os.path.join(sourcepath, "Kinetics-{}".format(n_classes), "annotations", "train.csv"), 'r') as f:
    #     data = [r[:-1].split(',') for r in f.readlines()][1:]
    # with open(os.path.join(sourcepath, "Kinetics-{}".format(n_classes), "annotations", "val.csv"), 'r') as f:
    #     data += [r[:-1].split(',') for r in f.readlines()][1:]

    with open(os.path.join(sourcepath, "Kinetics-{}".format(n_classes), "annotations", "meta.json"), "r") as fp:
        meta_dict = json.load(fp)

    exist_folders = glob.glob(os.path.join(sourcepath, "Kinetics-{}".format(n_classes), "frames", "*"))

    fnames, labels = [], []
    for folder in exist_folders:
        label = meta_dict[os.path.basename(folder)]
        fnames.append(folder)
        labels.append(label)

    classes = sorted(np.unique(labels).tolist())

    return fnames, labels, classes
"""========================================================="""


def filter_samples(opt, fnames, labels, classes):
    """
    Select a subset of classes. Mostly for faster debugging.
    """
    fnames, labels = np.array(fnames), np.array(labels)
    if opt.train_samples != -1:
        sel = np.linspace(0, len(fnames)-1, min(opt.train_samples, len(fnames))).astype(int)
        fnames, labels = fnames[sel], labels[sel]
    return np.array(fnames), np.array(labels), np.array(classes)


def filter_classes(opt, fnames, labels, classes, class_embedding):
    """
    Select a subset of classes. Mostly for faster debugging.
    """
    sel = np.ones(len(classes)) == 1
    if opt.class_total > 0:
        sel = np.linspace(0, len(classes)-1, opt.class_total).astype(int)

    classes = np.array(classes)[sel].tolist()
    class_embedding = class_embedding[sel]
    fnames = [f for i, f in enumerate(fnames) if labels[i] in classes]
    labels = [l for l in labels if l in classes]
    return np.array(fnames), np.array(labels), np.array(classes), class_embedding


def filter_overlapping_classes(fnames, labels, classes, class_embedding, ucf_class_embedding, class_overlap):
    class_distances = cdist(class_embedding, ucf_class_embedding, 'cosine').min(1)
    # sel = class_distances >= class_overlap
    sel = class_distances > class_overlap

    classes = np.array(classes)[sel].tolist()
    class_embedding = class_embedding[sel]

    fnames = [f for i, f in enumerate(fnames) if labels[i] in classes]
    labels = [l for l in labels if l in classes]

    return fnames, labels, classes, class_embedding


"""========================================================="""


def load_clips_tsn(fname, clip_len=16, n_clips=1, is_validation=False):
    if not os.path.exists(fname):
        print('Missing: '+fname)
        return []
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print('loading error, switching video ...')
        print(fname)
        return []

    total_frames = frame_count #min(frame_count, 300)
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])

    frames = []
    count = ret_count = 0
    while count < selection[-1]+clip_len:
        retained, frame = capture.read()
        if count not in selection:
            count += 1
            continue
        if not retained:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height, frame_width, 3)).astype('uint8')
            frames.append(frame)
            ret_count += 1
            count += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    capture.release()
    frames = np.stack(frames)
    total = n_clips * clip_len
    while frames.shape[0] < total:
        frames = np.concatenate([frames, frames[:(total - frames.shape[0])]])
    frames = frames.reshape([n_clips, clip_len, frame_height, frame_width, 3])
    return frames


def load_frames_tsn(fname, clip_len=16, n_clips=1, is_validation=False):
    if not os.path.exists(fname):
        print('Missing: '+fname)
        return []

    frame_count = len(glob.glob(os.path.join(fname, "images", "*")))
    one_frame = cv2.imread(os.path.join(fname, "images", "img_00001.jpg"))
    frame_height, frame_width, _ = one_frame.shape

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print('loading error, switching video ...')
        print(fname)
        return []

    total_frames = frame_count #min(frame_count, 300)
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])

    frames = []
    count = ret_count = 0
    while count < selection[-1]+clip_len:
        retained = count < frame_count
        if count not in selection:
            count += 1
            continue
        if not retained:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height, frame_width, 3)).astype('uint8')
            frames.append(frame)
            ret_count += 1
            count += 1
            continue
        frame = cv2.imread(os.path.join(fname, "images", "img_{:05d}.jpg".format(count + 1)))
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height, frame_width, 3)).astype('uint8')
        frames.append(frame)
        count += 1
    frames = np.stack(frames)
    total = n_clips * clip_len
    while frames.shape[0] < total:
        frames = np.concatenate([frames, frames[:(total - frames.shape[0])]])
    frames = frames.reshape([n_clips, clip_len, frame_height, frame_width, 3])
    return frames


class VideoDataset(Dataset):

    def __init__(self, fnames, labels, class_embed, classes, name, load_clips=load_clips_tsn,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False, evaluation_only=False):
        if 'kinetics' in name:
            fnames, labels = self.clean_data(fnames, labels)
        self.data = fnames
        self.labels = labels
        self.class_embed = class_embed
        self.class_name = classes
        self.name = name

        self.clip_len = clip_len
        self.n_clips = n_clips

        self.crop_size = crop_size  # 112
        self.is_validation = is_validation

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        self.transform = get_transform(self.is_validation, crop_size)
        self.loadvideo = load_clips
        split = 0
        # caption_folder = "/mnt/hdd1/captions"
        # self.image_caption_paths = \
        #     [os.path.join(caption_folder, "COCO", "captions_train2014.json"),
        #      os.path.join(caption_folder, "COCO", "captions_val2014.json")]
        # self.video_caption_paths = \
        #     [os.path.join(caption_folder, "ActivityNet", "train.json"),
        #      os.path.join(caption_folder, "ActivityNet", "val_1.json"),
        #      os.path.join(caption_folder, "ActivityNet", "val_2.json")]
        #
        # # wv_model = Word2Vec.load('./assets/GoogleNewsAdded', mmap='r')
        # # wv_model = wv_model.key_to_index
        #
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # model = AutoModel.from_pretrained("bert-base-uncased").cuda()
        #
        # bert_vocab = np.load("/mnt/hdd1/captions/bert_vocab.npy")
        #
        # vocab = np.zeros(dtype=np.float32, shape=(30000, 768))
        # counts = np.zeros(dtype=np.float32, shape=(30000, ))
        # image_captions = list()
        # UNK_count = 0
        # max_len = -1
        # for c_i, path in enumerate(self.image_caption_paths):
        #     with open(path, "r") as fp:
        #         caption_json = json.load(fp)
        #         for datum in tqdm(caption_json["annotations"], desc="Image Caption ({})".format(c_i + 1)):
        #             caption = datum["caption"]
        #
        #             inputs = tokenizer(caption, return_tensors="pt")
        #             for key in inputs.keys():
        #                 inputs[key] = inputs[key].cuda()
        #             outputs = model(**inputs)
        #             input_ids = inputs["input_ids"].detach().cpu().numpy().squeeze(0)
        #             embeddings = outputs["last_hidden_state"].detach().cpu().numpy().squeeze(0)
        #             # for i, this_id in enumerate(input_ids):
        #             #     vocab[this_id] += embeddings[i]
        #             #     counts[this_id] += 1.0
        #
        #             # l, c
        #             fake_samples = embeddings
        #             # l, vocab, c
        #             distances = np.square(np.expand_dims(bert_vocab, axis=0) - np.expand_dims(fake_samples, axis=1))
        #             # l, vocab
        #             distances = np.sum(distances, axis=-1)
        #             # l
        #             word_ids = np.argmin(distances, axis=-1)
        #
        #             sampled_word_ids = word_ids
        #             decoded_str = tokenizer.decode(sampled_word_ids.tolist())
        #             print(caption)
        #             print(decoded_str)
        #             exit()
        #
        #
        #             # if len(embeddings) < 50:
        #             #     embeddings = np.pad(embeddings, ((0, 50 - len(embeddings)), (0, 0)))
        #
        #             # tokens = self.preprocess_text(caption)
        #             # caption = self.clean_text(caption)
        #             # caption = self.clean_numbers(caption)
        #             # tokens = word_tokenize(caption)
        #             # tokens.append("<EOS>")
        #             # this_len = len(tokens)
        #             # if this_len > max_len:
        #             #     max_len = this_len
        #             # embeddings = list()
        #             # for token in tokens:
        #             #     try:
        #             #         embeds = wv_model[token]
        #             #     except KeyError:
        #             #         if token.lower() in wv_model:
        #             #             embeds = wv_model[token.lower()]
        #             #         elif token.lower().title() in wv_model:
        #             #             embeds = wv_model[token.lower().title()]
        #             #         elif "-" in token:
        #             #             new_tokens = token.split("-")
        #             #             for new_token in new_tokens:
        #             #                 try:
        #             #                     embeds = wv_model[new_token]
        #             #                 except KeyError:
        #             #                     if token.lower() in wv_model:
        #             #                         embeds = wv_model[token.lower()]
        #             #                     elif token.lower().title() in wv_model:
        #             #                         embeds = wv_model[token.lower().title()]
        #             #                     else:
        #             #                         embeds = wv_model["<UNK>"]
        #             #                         UNK_count += 1
        #             #         else:
        #             #             embeds = wv_model["<UNK>"]
        #             #             UNK_count += 1
        #             #     embeddings.append(embeds)
        #             # embeddings = np.array(embeddings, dtype=np.int64)
        #             # if len(embeddings) < 50:
        #             #     embeddings = np.concatenate((embeddings,
        #             #                                  np.zeros(dtype=np.float32, shape=(50 - len(embeddings), 300))),
        #             #                                 axis=0)
        #             image_captions.append(caption)
        # # np.save(os.path.join(caption_folder, "COCO", "image_captions.npy"), np.array(image_captions, dtype=np.float32))
        # print("Image Captions: {} Sentences, {} UNK, MAXLEN {}".format(len(image_captions), UNK_count, max_len))
        #
        # # with open(os.path.join(caption_folder, "COCO", "image_captions.json"), "w") as fp:
        # #     json.dump(image_captions, fp, indent=4, sort_keys=True)
        #
        # video_captions = list()
        # UNK_count = 0.0
        # max_len = -1
        # for c_i, path in enumerate(self.video_caption_paths):
        #     with open(path, "r") as fp:
        #         caption_json = json.load(fp)
        #         for identity in tqdm(caption_json.keys(), desc="Video Caption ({})".format(c_i + 1)):
        #             captions = caption_json[identity]["sentences"]
        #             for caption in captions:
        #                 inputs = tokenizer(caption, return_tensors="pt")
        #                 for key in inputs.keys():
        #                     inputs[key] = inputs[key].cuda()
        #
        #                 with torch.no_grad():
        #                     outputs = model(**inputs)
        #                 input_ids = inputs["input_ids"].detach().cpu().numpy().squeeze(0)
        #                 embeddings = outputs["last_hidden_state"].detach().cpu().numpy().squeeze(0)
        #                 for i, this_id in enumerate(input_ids):
        #                     vocab[this_id] += embeddings[i]
        #                     counts[this_id] += 1.0
        #                 # if len(embeddings) < 83:
        #                 #     embeddings = np.pad(embeddings, ((0, 83 - len(embeddings)), (0, 0)))
        #
        #                 # caption = self.clean_text(caption)
        #                 # caption = self.clean_numbers(caption)
        #                 # tokens = word_tokenize(caption)
        #                 # tokens.append("<EOS>")
        #                 # this_len = len(tokens)
        #                 # if this_len > max_len:
        #                 #     max_len = this_len
        #                 # embeddings = list()
        #                 # for token in tokens:
        #                 #     try:
        #                 #         embeds = wv_model[token]
        #                 #     except KeyError:
        #                 #         if token.lower() in wv_model:
        #                 #             embeds = wv_model[token.lower()]
        #                 #         elif token.lower().title() in wv_model:
        #                 #             embeds = wv_model[token.lower().title()]
        #                 #         elif "-" in token:
        #                 #             new_tokens = token.split("-")
        #                 #             for new_token in new_tokens:
        #                 #                 try:
        #                 #                     embeds = wv_model[new_token]
        #                 #                 except KeyError:
        #                 #                     if token.lower() in wv_model:
        #                 #                         embeds = wv_model[token.lower()]
        #                 #                     elif token.lower().title() in wv_model:
        #                 #                         embeds = wv_model[token.lower().title()]
        #                 #                     else:
        #                 #                         embeds = wv_model["<UNK>"]
        #                 #                         UNK_count += 1
        #                 #         else:
        #                 #             embeds = wv_model["<UNK>"]
        #                 #             UNK_count += 1
        #                 #     embeddings.append(embeds)
        #                 # embeddings = np.array(embeddings, dtype=np.int64)
        #                 # if len(embeddings) < 83:
        #                 #     embeddings = np.concatenate((embeddings,
        #                 #                                  np.zeros(dtype=np.float32, shape=(83 - len(embeddings), 300))),
        #                 #                                 axis=0)
        #                 video_captions.append(caption)
        # # np.save(os.path.join(caption_folder, "ActivityNet", "video_captions.npy"),
        # #         np.array(video_captions, dtype=np.float32))
        # print("Video Captions: {} Sentences, {} UNK, MAXLEN {}".format(len(video_captions), UNK_count, max_len))
        #
        # # with open(os.path.join(caption_folder, "ActivityNet", "video_captions.json"), "w") as fp:
        # #     json.dump(video_captions, fp, indent=4, sort_keys=True)
        #
        # vocab = vocab / np.expand_dims(np.maximum(counts, 1.0), axis=-1)
        # np.save(os.path.join(caption_folder, "bert_vocab.npy"), vocab)
        #
        # exit()
        split = 0
        if 'kinetics' in name:
            self.max_seq_len = 3
            caption_folder = "/mnt/hdd1/captions"
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            # self.model = AutoModel.from_pretrained("bert-base-uncased")
            with open(os.path.join(caption_folder, "COCO", "image_captions.json"), "r") as fp:
                self.image_captions = json.load(fp)
            with open(os.path.join(caption_folder, "ActivityNet", "video_captions.json"), "r") as fp:
                self.video_captions = json.load(fp)
            # self.image_captions = np.load(os.path.join(caption_folder, "COCO", "image_captions.npy"), allow_pickle=True)
            # self.video_captions = np.load(os.path.join(caption_folder, "ActivityNet", "video_captions.npz"))
            # self.image_captions = sorted(glob.glob(os.path.join(caption_folder, "COCO", "image_captions", "*.npy")))
            # self.video_captions = sorted(glob.glob(os.path.join(caption_folder, "ActivityNet", "video_captions", "*.npy")))

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label_array[idx]
        buffer = self.loadvideo(sample, self.clip_len, self.n_clips, self.is_validation)
        if len(buffer) == 0:
            buffer = np.random.rand(self.n_clips, 3, self.clip_len, 112, 112).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embed[0], -1
        s = buffer.shape
        buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        buffer = torch.stack([torch.from_numpy(im) for im in buffer], 0)
        buffer = self.transform(buffer)
        buffer = buffer.reshape(3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)

        if 'kinetics' in self.name:
            # image_caption = torch.Tensor(random.choice(self.image_captions)).long()
            # image_caption = F.one_hot(image_caption, 3000002).float()
            # image_caption = torch.Tensor(random.choice(self.image_captions)).float()
            # image_captions = random.sample(self.image_captions, 5)

            cap_ids = random.sample(range(len(self.image_captions)), 1)
            i_caption_embeddings = list()
            for cap_id in cap_ids:
                # image_caption = self.tokenizer(image_caption, return_tensors="pt")
                # with torch.no_grad():
                #     image_caption = self.model(**image_caption)
                # image_caption = image_caption["last_hidden_state"].detach().squeeze(0)
                image_caption = np.load(self.image_captions[cap_id])
                image_caption = torch.Tensor(image_caption)
                if len(image_caption) > self.max_seq_len:
                    random_start_index = random.choice(range(len(image_caption) - self.max_seq_len + 1))
                    image_caption = image_caption[random_start_index:random_start_index + self.max_seq_len]
                elif len(image_caption) < self.max_seq_len:
                    image_caption = F.pad(image_caption, (0, 0, 0, self.max_seq_len - len(image_caption)),
                                          "constant", value=0.0)
                i_caption_embeddings.append(image_caption)
            i_caption_embeddings = torch.stack(i_caption_embeddings, dim=0)

            cap_ids = random.sample(range(len(self.video_captions)), 1)
            v_caption_embeddings = list()
            for cap_id in cap_ids:
                # video_caption = self.video_captions["{}".format(cap_id)]
                video_caption = np.load(self.video_captions[cap_id])
                # video_caption = self.tokenizer(video_caption, return_tensors="pt")
                # with torch.no_grad():
                #     video_caption = self.model(**video_caption)
                # video_caption = video_caption["last_hidden_state"].detach().squeeze(0)
                video_caption = torch.Tensor(video_caption)
                if len(video_caption) > self.max_seq_len:
                    random_start_index = random.choice(range(len(video_caption) - self.max_seq_len + 1))
                    video_caption = video_caption[random_start_index:random_start_index + self.max_seq_len]
                elif len(video_caption) < self.max_seq_len:
                    video_caption = F.pad(video_caption, (0, 0, 0, self.max_seq_len - len(video_caption)),
                                          "constant", value=0.0)
                v_caption_embeddings.append(video_caption)
            v_caption_embeddings = torch.stack(v_caption_embeddings, dim=0)

            # image_captions = random.sample(self.image_captions, 1)
            # image_caption_tensors = list()
            # for image_caption in image_captions:
            #     image_caption = self.tokenizer(image_caption, return_tensors="pt")
            #     if len(image_caption["input_ids"][0]) > self.max_seq_len:
            #         random_start_index = random.choice(range(len(image_caption["input_ids"][0]) - self.max_seq_len + 1))
            #         for key in image_caption.keys():
            #             image_caption[key] = \
            #                 image_caption[key][:, random_start_index:random_start_index + self.max_seq_len]
            #     elif len(image_caption["input_ids"][0]) < self.max_seq_len:
            #         for key in image_caption.keys():
            #             image_caption[key] = F.pad(image_caption[key],
            #                                        (0, self.max_seq_len - len(image_caption[key][0])),
            #                                        "constant", value=0)
            #     image_caption_tensors.append(image_caption)
            # # image_caption_tensors = torch.stack(image_caption_tensors, dim=0)
            #
            # video_captions = random.sample(self.video_captions, 1)
            # video_caption_tensors = list()
            # for video_caption in video_captions:
            #     video_caption = self.tokenizer(video_caption, return_tensors="pt")
            #     if len(video_caption["input_ids"][0]) > self.max_seq_len:
            #         random_start_index = random.choice(range(len(video_caption["input_ids"][0]) - self.max_seq_len + 1))
            #         for key in video_caption.keys():
            #             video_caption[key] = \
            #                 video_caption[key][:, random_start_index:random_start_index + self.max_seq_len]
            #     elif len(video_caption["input_ids"][0]) < self.max_seq_len:
            #         for key in video_caption.keys():
            #             video_caption[key] = F.pad(video_caption[key],
            #                                        (0, self.max_seq_len - len(video_caption[key][0])),
            #                                        "constant", value=0)
            #     video_caption_tensors.append(video_caption)
            # # video_caption_tensors = torch.stack(video_caption_tensors, dim=0)

            return buffer, label, self.class_embed[label], idx, (i_caption_embeddings, v_caption_embeddings)
        else:
            return buffer, label, self.class_embed[label], idx

    def __len__(self):
        return len(self.data)

    @staticmethod
    def clean_data(fnames, labels):
        if not isinstance(fnames[0], str):
            print('Cannot check for broken videos')
            return fnames, labels
        broken_videos_file = 'assets/kinetics_broken_videos.txt'
        if not os.path.exists(broken_videos_file):
            print('Broken video list does not exists')
            return fnames, labels

        t = time()
        with open(broken_videos_file, 'r') as f:
            broken_samples = [r[:-1] for r in f.readlines()]
        data = [x[75:] for x in fnames]
        keep_sample = np.in1d(data, broken_samples) == False
        fnames = np.array(fnames)[keep_sample]
        labels = np.array(labels)[keep_sample]
        print('Broken videos %.2f%% - removing took %.2f' % (100 * (1.0 - keep_sample.mean()), time() - t))
        return fnames, labels

    # preprocess the text.
    def preprocess_text(self, text):
        mystopwords = set(stopwords.words("english"))

        def remove_stops_digits(tokens):
            # Nested function that lowercases, removes stopwords and digits from a list of tokens
            return [token.lower() for token in tokens if token.lower() not in mystopwords and not token.isdigit()
                    and token not in punctuation]

        # This return statement below uses the above function to process twitter tokenizer output further.
        return remove_stops_digits(word_tokenize(text))

    def clean_text(self, x):
        pattern = r'[^a-zA-z0-9\s]'
        x = re.sub(pattern, '', x)
        return x

    def clean_numbers(self, x):
        if bool(re.search(r'\d', x)):
            x = re.sub('[0-9]{5,}', '#####', x)
            x = re.sub('[0-9]{4}', '####', x)
            x = re.sub('[0-9]{3}', '###', x)
            x = re.sub('[0-9]{2}', '##', x)
        return x

