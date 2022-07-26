############################# LIBRARIES ######################################
import torch
import torch.nn as nn
# import torchvision.models as models
import resnet as models
import torch.nn.functional as F
import numpy as np
from gensim.models import KeyedVectors as Word2Vec
from tqdm import tqdm

import sys
from PIL import Image
sys.path.append(r'../MAGIC/image_captioning/language_model/')
sys.path.append(r'../MAGIC/image_captioning/clip/')

from simctg import SimCTG
from clip import CLIP

"""=================================================================================================================="""


def get_network(opt):
    """
    Selection function for available networks.
    """
    if 'r3d' in opt.network:
        # network = models.video.r3d_18
        network = models.r3d_18
    elif '2plus1d' in opt.network:
        # network = models.video.r2plus1d_18
        network = models.r2plus1d_18
    elif 'c3d' in opt.network:
        return C3D(fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)
    else:
        raise Exception('Network {} not available!'.format(opt.network))
    model = Model(network, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)
    # decoder = Decoder()
    # encoder = Encoder()
    # return ResNet18(network, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)
    # return Model(network, decoder=decoder, encoder=encoder, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)

    return model


"""=================================================================================================================="""


class ResNet18(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, network, fixconvs=False, nopretrained=True):
        super(ResNet18, self).__init__()
        self.model = network(pretrained=nopretrained)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False

        self.regressor = nn.Linear(self.model.fc.in_features, 300)
        self.dropout = torch.nn.Dropout(p=0.05)
        # self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        # model.fc.weight.requires_grad = True
        # model.fc.bias.requires_grad = True

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)
        x, f = self.model(x)
        x = x.view(bs*nc, -1)
        x = x.reshape(bs, nc, -1)
        x = torch.mean(x, 1)
        x = self.dropout(x)
        x = self.regressor(x)
        x = F.normalize(x)
        return x


"""=================================================================================================================="""

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.

C3D code taken from: https://github.com/DavideA/c3d-pytorch
"""


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, fixconvs=False, nopretrained=True):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        if nopretrained:
            self.load_state_dict(torch.load('./assets/c3d.pickle'))

        self.regressor = nn.Linear(4096, 300)

        if fixconvs:
            for model in [self.conv1, self.conv2,
                          self.conv3a, self.conv3b,
                          self.conv4a, self.conv4b,
                          self.conv5a, self.conv5b,
                          self.fc6]:
                for param in model.parameters():
                    param.requires_grad = False

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        # h = self.relu(self.fc7(h))
        # h = self.dropout(h)

        # logits = self.fc8(h)
        # probs = self.softmax(logits)

        h = h.reshape(bs, nc, -1)
        h = torch.mean(h, 1)
        h = h.reshape(bs, -1)

        h = self.regressor(h)
        h = torch.nn.functional.normalize(h, dim=-1)
        return h


"""=================================================================================================================="""


# class Model(nn.Module):
#     """
#     Container for ResNet50 s.t. it can be used for metric learning.
#     The Network has been broken down to allow for higher modularity, if one wishes
#     to target specific layers/blocks directly.
#     """
#
#     def __init__(self, network, fixconvs=False, nopretrained=False):
#         super(Model, self).__init__()
#         self.model = network(pretrained=nopretrained)
#         if fixconvs:
#             for param in self.model.parameters():
#                 param.requires_grad = False
#
#         # self.decoder = decoder()
#         # self.encoder = encoder()
#
#         self.regressor = nn.Linear(self.model.fc.in_features, 300)
#         self.dropout = torch.nn.Dropout(p=0.05)
#
#     def forward(self, x, real_samples=None):
#         bs, nc, ch, l, h, w = x.shape
#         x = x.reshape(bs*nc, ch, l, h, w)
#         x, f = self.model(x)
#
#         x = self.dropout(x)
#         x = self.regressor(x)
#         x = F.normalize(x)
#
#         # # bs, l, v
#         # fake_samples = self.decoder(f)
#         #
#         # (fake_dis_01, fake_dis_02), fake_emb = self.encoder(fake_samples, twice=True, embed=True)
#         # if real_samples is not None:
#         #     (real_dis, _), _ = self.encoder(real_samples, twice=False, embed=False)
#         # else:
#         #     real_dis = None
#         #
#         # return fake_samples, fake_emb, (real_dis, (fake_dis_01, fake_dis_02))
#
#         return x, f
#
#
# class Decoder(nn.Module):
#
#     def __init__(self):
#         super(Decoder, self).__init__()
#
#         self.d_model = 256
#         self.temperature = 1.0
#         self.max_seq_len = 20
#
#         # self.wv_model = Word2Vec.load('./assets/GoogleNewsAdded', mmap='r')
#         split = 0
#         # self.embeddings = list()
#         # for w_i in tqdm(range(len(self.wv_model))):
#         #     self.embeddings.append(self.wv_model[self.wv_model.index_to_key[w_i]])
#         # embeddings = np.array(self.embeddings, dtype=np.float32)
#         # np.save("./assets/embeddings.npy", embeddings)
#         # exit()
#         split = 0
#         # self.embeddings = np.load("./assets/embeddings.npy")
#         # self.embeddings = np.zeros(dtype=np.float32, shape=(3000002, 300))
#         # self.embeddings = torch.Tensor(self.embeddings).cuda()
#         self.t_pos_embeds = nn.Embedding(2, self.d_model)
#         self.h_pos_embeds = nn.Embedding(7, self.d_model)
#         self.w_pos_embeds = nn.Embedding(7, self.d_model)
#         self.s_pos_embeds = nn.Embedding(self.max_seq_len, self.d_model)
#
#         # self.word2input_proj = nn.Linear(300, self.d_model)
#         self.feature2input_proj = nn.Linear(512, self.d_model)
#         decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, dim_feedforward=self.d_model * 4,
#                                                    nhead=8, dropout=0.1, activation="gelu")
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
#         self.output2word_proj = nn.Linear(self.d_model, 30000)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#         nn.init.normal_(self.t_pos_embeds.weight)
#         nn.init.normal_(self.h_pos_embeds.weight)
#         nn.init.normal_(self.w_pos_embeds.weight)
#         nn.init.normal_(self.s_pos_embeds.weight)
#
#     def forward(self, feats):
#         """
#         RelGAN step forward
#         :param inp: [batch_size]
#         :param hidden: memory size
#         :return: pred, hidden, next_token, next_token_onehot, next_o
#             - pred: batch_size * vocab_size, use for adversarial training backward
#             - hidden: next hidden
#             - next_token: [batch_size], next sentence token
#             - next_token_onehot: batch_size * vocab_size, not used yet
#             - next_o: batch_size * vocab_size, not used yet
#         """
#         bs, c, t, h, w = feats.shape
#         feats = self.feature2input_proj(feats.view(bs, c, t * h * w).permute(0, 2, 1))
#         pos_embeds = (self.t_pos_embeds.weight.view(t, 1, 1, self.d_model) +
#                       self.h_pos_embeds.weight.view(1, h, 1, self.d_model) +
#                       self.w_pos_embeds.weight.view(1, 1, w, self.d_model)).view(1, t * h * w, self.d_model)
#         feats = feats + pos_embeds.cuda()
#
#         s_pos_embeds = self.s_pos_embeds.weight.view(1, self.max_seq_len, self.d_model).repeat(bs, 1, 1).cuda()
#         out = self.decoder(s_pos_embeds.permute(1, 0, 2), feats.permute(1, 0, 2)).permute(1, 0, 2)
#         out = self.output2word_proj(out)
#         out = F.gumbel_softmax(out, tau=1, hard=True)
#
#         return out
#
#     def step(self, embs, feats):
#         """
#         RelGAN step forward
#         :param inp: [batch_size]
#         :param hidden: memory size
#         :return: pred, hidden, next_token, next_token_onehot, next_o
#             - pred: batch_size * vocab_size, use for adversarial training backward
#             - hidden: next hidden
#             - next_token: [batch_size], next sentence token
#             - next_token_onehot: batch_size * vocab_size, not used yet
#             - next_o: batch_size * vocab_size, not used yet
#         """
#         out = self.decoder(embs.permute(1, 0, 2), feats.permute(1, 0, 2)).permute(1, 0, 2)
#         out = self.output2word_proj(out[:, -1])
#
#         # pred = F.gumbel_softmax(out, tau=self.temperature, hard=True, dim=-1)
#         # next_token = torch.argmax(pred, dim=1).detach()
#
#         gumbel_t = self.add_gumbel(out)
#         next_token = torch.argmax(gumbel_t, dim=1).detach()
#
#         pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size
#
#         return pred, next_token
#
#     def sample(self, feats, start_letter="</s>", end_letter="<EOS>"):
#         """
#         Sample from RelGAN Generator
#         - one_hot: if return pred of RelGAN, used for adversarial training
#         :return:
#             - all_preds: batch_size * seq_len * vocab_size, only use for a batch
#             - samples: all samples
#         """
#         bs, c, t, h, w = feats.shape
#         feats = self.feature2input_proj(feats.view(bs, c, t * h * w).permute(0, 2, 1))
#         pos_embeds = (self.t_pos_embeds.weight.view(t, 1, 1, self.d_model) +
#                       self.h_pos_embeds.weight.view(1, h, 1, self.d_model) +
#                       self.w_pos_embeds.weight.view(1, 1, w, self.d_model)).view(1, t * h * w, self.d_model)
#         feats = feats + pos_embeds.cuda()
#
#         s_pos_embeds = self.s_pos_embeds.weight.view(1, self.max_seq_len, self.d_model).cuda()
#
#         # all_preds = list()
#         all_preds = torch.zeros(bs, self.max_seq_len, len(self.wv_model)).cuda()
#         all_samples = list()
#
#         embeddings = self.wv_model[start_letter]
#         inp = torch.Tensor([embeddings] * bs).view(bs, 1, 300).cuda().detach()
#         inp = self.word2input_proj(inp)
#
#         end_flags = np.asarray([False] * bs)
#         for i in range(self.max_seq_len):
#             pred, next_token = self.step(inp, feats)
#             next_token = np.asarray([self.wv_model.index_to_key[idx] for idx in next_token.cpu().numpy().tolist()])
#             # pred_embeddings = torch.matmul(pred, self.embeddings)
#             # pred_embeddings[end_flags] = torch.zeros_like(pred_embeddings[0])
#             # pred[end_flags] = torch.zeros_like(pred[0])
#             # all_preds.append(pred)
#             all_preds[np.logical_not(end_flags), i] = pred[np.logical_not(end_flags)]
#             next_inp = torch.Tensor(self.wv_model[next_token]).view(bs, 300).detach().cuda()
#             next_inp[end_flags] = torch.zeros_like(next_inp[0])
#             inp = torch.cat((inp, (self.word2input_proj(next_inp) + s_pos_embeds[:, i]).unsqueeze(1)), dim=1)
#             # inp = torch.cat((inp, (self.word2input_proj(pred_embeddings) + s_pos_embeds[:, i]).unsqueeze(1)), dim=1)
#             next_token[end_flags] = ""
#             all_samples.append(next_token)
#             end_flags = np.logical_or(end_flags, next_token == end_letter)
#             print(i)
#         # all_preds = torch.stack(all_preds, dim=1)
#         all_samples = np.stack(all_samples, axis=1).tolist()
#
#         return all_preds, all_samples
#
#     @staticmethod
#     def add_gumbel(o_t, eps=1e-10):
#         """Add o_t by a vector sampled from Gumbel(0,1)"""
#         u = torch.zeros(o_t.size()).cuda()
#
#         u.uniform_(0, 1)
#         g_t = -torch.log(-torch.log(u + eps) + eps)
#         gumbel_t = o_t + g_t
#         return gumbel_t
#
#
# class Encoder(nn.Module):
#
#     def __init__(self):
#         super(Encoder, self).__init__()
#
#         self.d_model = 128
#         self.max_seq_len = 20
#
#         # self.wv_model = Word2Vec.load('./assets/GoogleNewsAdded', mmap='r')
#         self.s_pos_embeds = nn.Embedding(self.max_seq_len, self.d_model)
#
#         self.special_tokens = nn.Embedding(2, self.d_model)
#
#         self.word2input_proj = nn.Linear(768, self.d_model)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward=self.d_model * 4,
#                                                    nhead=8, dropout=0.1, activation="gelu")
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.output2dis_proj = nn.Linear(self.d_model, 1)
#         self.output2emb_proj = nn.Linear(self.d_model, 300)
#         # self.output2dis_proj = MLP(self.d_model, self.d_model, 1, 3)
#         # self.output2emb_proj = MLP(self.d_model, self.d_model, 300, 3)
#
#         self.dropout = torch.nn.Dropout(p=0.50)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#         nn.init.normal_(self.s_pos_embeds.weight)
#         nn.init.xavier_uniform_(self.special_tokens.weight)
#
#     def forward(self, x, embed=True, twice=False):
#         bs = x.shape[0]
#
#         special_tokens = self.special_tokens.weight.view(1, 2, self.d_model).repeat(bs, 1, 1).cuda()
#         s_pos_embeds = self.s_pos_embeds.weight.view(1, self.max_seq_len, self.d_model).cuda()
#         out = self.word2input_proj(x) + s_pos_embeds
#         # if not embed:
#         #     out = self.dropout(out)
#         out = torch.cat((special_tokens, out), dim=1)
#
#         out = self.encoder(out.permute(1, 0, 2)).permute(1, 0, 2)
#         dis_out_01 = self.output2dis_proj(out[:, 0]).squeeze(-1)
#         if embed:
#             emb_out = F.normalize(self.output2emb_proj(out[:, 1]))
#         else:
#             emb_out = None
#
#         # if twice:
#         #     special_tokens = self.special_tokens.weight.view(1, 2, self.d_model).repeat(bs, 1, 1).cuda()
#         #     s_pos_embeds = self.s_pos_embeds.weight.view(1, self.max_seq_len, self.d_model).cuda()
#         #     out = self.word2input_proj(x.detach()) + s_pos_embeds
#         #     # out = self.dropout(out)
#         #     out = torch.cat((special_tokens, out), dim=1)
#         #
#         #     out = self.encoder(out.permute(1, 0, 2)).permute(1, 0, 2)
#         #     dis_out_02 = self.output2dis_proj(out[:, 0]).squeeze(-1)
#         # else:
#         #     dis_out_02 = None
#         #
#         # return (dis_out_01, dis_out_02), emb_out
#
#         return emb_out, dis_out_01
#
#
# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""
#
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, last_activate=False):
#         super().__init__()
#         self.num_layers = num_layers
#         self.last_activate = last_activate
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
#
#     def forward(self, x):
#         # x = x.permute(0, 2, 1)
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         # x = x.permute(0, 2, 1)
#         return x


"""=================================================================================================================="""


class Model(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, network, fixconvs=False, nopretrained=False):
        super(Model, self).__init__()
        self.model = network(pretrained=nopretrained)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False

        # Load Language Model
        language_model_name = r'cambridgeltl/magic_mscoco'  # or r'/path/to/downloaded/cambridgeltl/magic_mscoco'
        self.sos_token, self.pad_token = r'<-start_of_text->', r'<-pad->'
        self.k, self.alpha, self.beta, self.decoding_len = 45, 0.1, 2.0, 16
        self.generation_model = SimCTG(language_model_name, self.sos_token, self.pad_token).cuda()

        model_name = r"openai/clip-vit-base-patch32"  # or r"/path/to/downloaded/openai/clip-vit-base-patch32"
        self.clip = CLIP(model_name).cuda()

        self.d_model = 128
        self.num_sentences = 4
        self.max_seq_len = self.decoding_len
        self.t_pos_embeds = nn.Embedding(2, self.d_model)
        self.h_pos_embeds = nn.Embedding(7, self.d_model)
        self.w_pos_embeds = nn.Embedding(7, self.d_model)
        self.s_pos_embeds = nn.Embedding(16, self.d_model)
        self.l_pos_embeds = nn.Embedding(self.max_seq_len, self.d_model)
        self.special_tokens = nn.Embedding(1, self.d_model)
        self.word2input_proj = nn.Linear(768, self.d_model)
        self.feature2input_proj = nn.Linear(512, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward=self.d_model * 4,
                                                   nhead=8, dropout=0.1, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.output2emb_proj = nn.Linear(self.d_model, 300)

        self.reset_parameters()

    def reset_parameters(self):
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        nn.init.normal_(self.t_pos_embeds.weight)
        nn.init.normal_(self.h_pos_embeds.weight)
        nn.init.normal_(self.w_pos_embeds.weight)
        nn.init.normal_(self.s_pos_embeds.weight)
        nn.init.normal_(self.l_pos_embeds.weight)
        nn.init.xavier_uniform_(self.special_tokens.weight)

    def forward(self, x):
        bs, nc, i_c, i_t, i_h, i_w = x.shape
        x = x.reshape(bs * nc, i_c, i_t, i_h, i_w)
        _, cnn_feats = self.model(x)

        _, v_c, v_t, v_h, v_w = cnn_feats.shape
        cnn_feats = self.feature2input_proj(cnn_feats.view(bs, v_c, v_t * v_h * v_w).permute(0, 2, 1))
        v_pos_embeds = (self.t_pos_embeds.weight.view(v_t, 1, 1, self.d_model) +
                        self.h_pos_embeds.weight.view(1, v_h, 1, self.d_model) +
                        self.w_pos_embeds.weight.view(1, 1, v_w, self.d_model)).view(1, v_t * v_h * v_w, self.d_model)
        cnn_feats = cnn_feats + v_pos_embeds.cuda()

        self.clip.eval()
        self.generation_model.eval()
        with torch.no_grad():
            start_token = self.generation_model.tokenizer.tokenize(self.sos_token)
            start_token_id = self.generation_model.tokenizer.convert_tokens_to_ids(start_token)
            input_ids = torch.LongTensor(start_token_id).unsqueeze(0).repeat(bs, 1).cuda()

            word_feats = list()
            word_samples = list()
            images = ((x.permute(0, 2, 3, 4, 1).detach().cpu().numpy() * 2 + 1) * 255.0).astype(np.uint8)
            image_indices = np.linspace(0, i_t, self.num_sentences, dtype=np.int32)
            for image_index in image_indices:
                # batch_word_feats = list()
                # for batch_index in range(bs):
                #     input_ids = torch.LongTensor(start_token_id).view(1, -1).cuda()
                #     image_instance = Image.fromarray(images[batch_index, image_index])
                #     w_feats, tokens = \
                #         self.generation_model.magic_search(input_ids, self.k, self.alpha, self.decoding_len,
                #                                            self.beta, image_instance, self.clip, 60)
                #     w_feats = self.word2input_proj(w_feats)
                #     print(w_feats.shape)
                #     batch_word_feats.append(w_feats)
                # batch_word_feats = torch.cat(batch_word_feats, dim=0)
                # word_feats.append(batch_word_feats)

                image_instance = images[:, image_index].unstack(axis=0)
                w_feats, tokens = \
                    self.generation_model.magic_search(input_ids, self.k, self.alpha, self.decoding_len,
                                                       self.beta, image_instance, self.clip, 60)
                w_feats = self.word2input_proj(w_feats)
                word_feats.append(w_feats)

                batch_word_samples = list()
                for this_tokens in tokens.unbind(dim=0):
                    text = self.generation_model.tokenizer.decode(this_tokens).strip()
                    text = ' '.join(text.split()).strip()
                    batch_word_samples.append(text)
                word_samples.append(batch_word_samples)
            word_feats = torch.cat(word_feats, dim=1)
            _, w_s, w_l, w_c = word_feats.shape
            w_pos_embeds = (self.s_pos_embeds.weight.view(w_s, 1, self.d_model) +
                            self.l_pos_embeds.weight.view(1, w_l, self.d_model)).view(1, w_s * w_l, self.d_model)
            word_feats = (word_feats + w_pos_embeds.cuda()).detach()

        special_tokens = self.special_tokens.weight.unsqueeze(0).repeat(bs, 1, 1).cuda()

        feats = torch.cat((special_tokens, cnn_feats, word_feats), dim=1)
        out = self.encoder(feats.permute(1, 0, 2)).permute(1, 0, 2)
        emb_out = F.normalize(self.output2emb_proj(out[:, 0]))

        return emb_out, word_samples


"""=================================================================================================================="""

if __name__ == "__main__":
    # decoder = Decoder
    # encoder = Encoder
    model = Model(network=models.r2plus1d_18, fixconvs=False, nopretrained=True).cuda()

    dummy_data = torch.tensor(np.zeros(dtype=np.float32, shape=(8, 1, 3, 16, 112, 112))).cuda()
    # dummy_captions = torch.Tensor(np.zeros(dtype=np.float32, shape=(8, 20, 768))).cuda()

    # # bs, l, v
    # fake_samples = decoder(dummy_data)
    #
    # adversarial_criterion = torch.nn.BCEWithLogitsLoss().cuda()
    #
    # fake_dis, fake_emb = encoder(fake_samples)
    # real_dis, real_emb = encoder(dummy_captions)
    #
    # d_loss = adversarial_criterion(real_dis - fake_dis, torch.ones_like(real_dis))
    # g_loss = adversarial_criterion(fake_dis - real_dis, torch.ones_like(fake_dis))
    # adv_loss = g_loss + d_loss
    # fake_dis.retain_grad()
    # fake_samples.retain_grad()
    # dummy_data.retain_grad()
    # adv_loss.backward()
    # print(fake_dis.grad)
    # print(fake_samples.grad)
    # print(dummy_data.grad)

    embed_criterion = torch.nn.MSELoss().cuda()
    # adversarial_criterion = torch.nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    # fake_emb, (real_dis, (fake_dis_01, fake_dis_02)) = model(dummy_data, dummy_captions)

    # f = cnn(dummy_data)
    #
    # fake_samples = decoder(f)
    # fake_dis, fake_emb = encoder(fake_samples)
    # real_dis, _ = encoder(dummy_captions)

    embeds, samples = model(dummy_data)

    print(samples)

    embed_loss = embed_criterion(embeds, torch.zeros_like(embeds))
    optimizer.zero_grad()
    embed_loss.backward()
    optimizer.step()

    # g_loss = adversarial_criterion(fake_dis_01 - real_dis.detach(), torch.ones_like(fake_dis_01))
    #
    # optimizer.zero_grad()
    # gan_optimizer.zero_grad()
    # g_loss.backward(retain_graph=True)
    # dis_optimizer.zero_grad()
    # embed_loss.backward()
    # # optimizer.step()
    # # gan_optimizer.step()
    # # dis_optimizer.step()
    #
    # print("embed loss done")
    #
    # # Compute loss.
    # # g_loss = adversarial_criterion(fake_dis - real_dis.detach(), torch.ones_like(fake_dis))
    # # adv_loss = g_loss + d_loss
    #
    # # optimizer.zero_grad()
    # # gan_optimizer.zero_grad()
    # # g_loss.backward()
    # # optimizer.step()
    # # gan_optimizer.step()
    #
    # print("gan loss done")
    #
    # # fake_emb, (real_dis, fake_dis) = model(dummy_data, dummy_captions)
    #
    # # fake_dis, _ = encoder(fake_samples.detach())
    # # real_dis, _ = encoder(dummy_captions)
    #
    # d_loss = adversarial_criterion(real_dis - fake_dis_02, torch.ones_like(real_dis))
    #
    # # dis_optimizer.zero_grad()
    # d_loss.backward()
    # optimizer.step()
    # gan_optimizer.step()
    # dis_optimizer.step()
    #
    # print("dis loss done")