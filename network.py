############################# LIBRARIES ######################################
import torch
import torch.nn as nn
# import torchvision.models as models
import resnet as models
import torch.nn.functional as F
from gensim.models import KeyedVectors as Word2Vec

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
    return ResNet18(network, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)


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


class Model(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, network, decoder, encoder, fixconvs=False, nopretrained=False):
        super(Model, self).__init__()
        self.model = network(pretrained=nopretrained)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False

        self.regressor = nn.Linear(self.model.fc.in_features, 300)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        # model.fc.weight.requires_grad = True
        # model.fc.bias.requires_grad = True

        self.d_model = 256
        self.word2input_proj = nn.Linear(300, self.d_model)
        self.feature2input_proj = nn.Linear(512, self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, dim_feedforward=self.d_model * 4,
                                                   nhead=8, dropout=0.1, activation="gelu")
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.output2word_proj = nn.Linear(self.d_model, 3000000)

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)
        x, f = self.model(x)

        f = self.feature2input_proj(f)


        x = x.view(bs*nc, -1)
        x = x.reshape(bs, nc, -1)
        x = torch.mean(x, 1)
        x = self.dropout(x)
        x = self.regressor(x)
        x = F.normalize(x)
        return x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.d_model = 256
        self.temperature = 1.0

        self.embeddings = Word2Vec.load('./assets/GoogleNews', mmap='r')
        self.t_pos_embeds = nn.Embedding(3, self.d_model)
        self.h_pos_embeds = nn.Embedding(4, self.d_model)
        self.w_pos_embeds = nn.Embedding(4, self.d_model)

        self.word2input_proj = nn.Linear(300, self.d_model)
        self.feature2input_proj = nn.Linear(512, self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, dim_feedforward=self.d_model * 4,
                                                   nhead=8, dropout=0.1, activation="gelu")
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.output2word_proj = nn.Linear(self.d_model, len(self.embeddings))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.normal_(self.t_pos_embeds.weight)
        nn.init.normal_(self.h_pos_embeds.weight)
        nn.init.normal_(self.w_pos_embeds.weight)

    def step(self, embs, feats):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        out = self.decoder(embs, feats)
        out = self.output2word_proj(out)
        gumbel_t = self.add_gumbel(out)
        next_token = torch.argmax(gumbel_t, dim=1).detach()

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size

        return pred, next_token

    def sample(self, feats, start_letter="</s>", end_letter="<EOS>"):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        bs, c, t, h, w = feats.size[0]
        feats = self.feature2input_proj(feats.view(bs, c, t * h * w)).permute(2, 0, 1)
        pos_embeds = (self.t_pos_embeds.weight.view(t, 1, 1, self.d_model) +
                      self.h_pos_embeds.weight.view(1, h, 1, self.d_model) +
                      self.w_pos_embeds.weight.view(1, 1, w, self.d_model)).view(1, t * h * w, self.d_model)
        feats = feats + pos_embeds
        all_preds = torch.zeros(bs, self.max_seq_len, self.vocab_size).cuda()
        all_preds = all_preds.cuda()

        embeddings = self.embeddings[start_letter]
        inp = torch.Tensor([embeddings] * bs).view(1, bs, 300).cuda()
        inp = inp.cuda()

        end_flags = [False] * bs
        for i in range(self.max_seq_len):
            pred, next_token = self.step(inp, feats)
            next_token = self.embeddings.index_to_key[next_token]
            all_preds[:, i] = pred
            next_inp = torch.Tensor(self.embeddings[next_token]).view(1, bs, 300).cuda()
            next_inp[end_flags] = torch.zeros_like(next_inp[:, 0])
            inp = torch.cat((inp, next_inp), dim=0)

            end_flags = next_token == end_letter

        return all_preds

    @staticmethod
    def add_gumbel(o_t, eps=1e-10):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size()).cuda()

        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t


"""=================================================================================================================="""

