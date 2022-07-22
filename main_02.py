import os, numpy as np, argparse, time, multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import network
import dataset
from auxiliary.transforms import batch2gif

from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

from colorama import Fore, Style
from torch.cuda.amp import GradScaler, autocast
import random
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

Style.RESET_ALL

"""=========================INPUT ARGUMENTS====================="""

parser = argparse.ArgumentParser()

parser.add_argument('--split',        default=-1,   type=int, help='Train/test classes split. Use -1 for kinetics2ucf')
parser.add_argument('--dataset',      default='kinetics2oboth',   type=str, help='Dataset: [kinetics2oboth, kinetics2others, sun2both]')

parser.add_argument('--train_samples',  default=-1,  type=int, help='Reduce number of train samples to the given value')
parser.add_argument('--class_total',  default=60,  type=int, help='For debugging only. Reduce the total number of classes')

parser.add_argument('--clip_len',     default=16,   type=int, help='Number of frames of each sample clip')
parser.add_argument('--n_clips',     default=1,   type=int, help='Number of clips per video')

parser.add_argument('--class_overlap', default=0.040,  type=float, help='tau. see Eq.3 in main paper')

### General Training Parameters
parser.add_argument('--lr',           default=1e-3, type=float, help='Learning Rate for network parameters.')
parser.add_argument('--n_epochs',     default=150,   type=int,   help='Number of training epochs.')
parser.add_argument('--bs',           default=64,   type=int,   help='Mini-Batchsize size per GPU.')
parser.add_argument('--size',         default=112,  type=int,   help='Image size in input.')

parser.add_argument('--fixconvs', action='store_true', default=False,   help='Freezing conv layers')
parser.add_argument('--nopretrained', action='store_false', default=False,   help='Pretrain network.')

##### Network parameters
parser.add_argument('--network', default='r2plus1d_18', type=str,
                    help='Network backend choice: [resnet18, r2plus1d_18, r3d_18, c3d].')

### Paths to datasets and storage folder
parser.add_argument('--save_path',    default='./experiments', type=str, help='Where to save log and checkpoint.')
parser.add_argument('--weights',      default=None, type=str, help='Weights to load from a previously run.')
parser.add_argument('--progressbar', action='store_true', default=True,   help='Show progress bar during train/test.')
parser.add_argument('--evaluate', action='store_true', default=False,   help='Evaluation only using 25 clips per video')

##### Read in parameters
opt = parser.parse_args()

opt.multiple_clips = False
opt.kernels = multiprocessing.cpu_count()
# opt.kernels = 10

"""=================================DATALOADER SETUPS====================="""
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    opt.bs = opt.bs * torch.cuda.device_count()

print('Total batch size: %d' % opt.bs)
dataloaders = dataset.get_datasets(opt)
if not opt.evaluate:
    opt.n_classes = dataloaders['training'][0].dataset.class_embed.shape[0]
else:
    opt.n_classes = dataloaders['testing'][0].dataset.class_embed.shape[0]

"""=================================OUTPUT FOLDER====================="""
opt.savename = opt.save_path + '/'
if not opt.evaluate:
    opt.savename += '%s/CLIP%d_LR%f_%s_BS%d' % (
            opt.dataset, opt.clip_len,
            opt.lr, opt.network, opt.bs)

    if opt.class_overlap > 0:
        opt.savename += '_CLASSOVERLAP%.2f' % opt.class_overlap

    if opt.class_total != -1:
        opt.savename += '_NCLASS%d' % opt.class_total

    if opt.train_samples != -1:
        opt.savename += '_NTRAIN%d' % opt.train_samples

    if opt.fixconvs:
        opt.savename += '_FixedConvs'

    if not opt.nopretrained:
        opt.savename += '_NotPretrained'

    count = 1
    while os.path.exists(opt.savename):
        opt.savename += '_{}'.format(count)
        count += 1

    if opt.split != -1:
        opt.savename += '/split%d' % opt.split

else:
    opt.weights = opt.savename + 'checkpoint.pth.tar'
    opt.savename += '/evaluation/'


if not os.path.exists(opt.savename+'/samples/'):
    os.makedirs(opt.savename+'/samples/')

"""=============================NETWORK SETUP==============================="""
opt.device = torch.device('cuda')
# model      = network.get_network(opt)
cnn, decoder, encoder = network.get_network(opt)

# cnn = model.model
# decoder = model.decoder
# encoder = model.encoder

# if opt.weights and opt.weights != "none":
#     #model.load_state_dict(torch.load(opt.weights)['state_dict'])
#     j = len('module.')
#     weights = torch.load(opt.weights)['state_dict']
#     model_dict = model.state_dict()
#     weights = {k[j:]: v for k, v in weights.items() if k[j:] in model_dict.keys()}
#     # if not opt.evaluate:
#     #     weights = {k: v for k, v in weights.items() if 'regressor' not in k}
#     model_dict.update(weights)
#     model.load_state_dict(model_dict)
#     print("LOADED MODEL:  ", opt.weights)

# model = nn.DataParallel(model)
# _ = model.to(opt.device)

cnn = nn.DataParallel(cnn).cuda()
decoder = nn.DataParallel(decoder).cuda()
encoder = nn.DataParallel(encoder).cuda()

"""==========================OPTIM SETUP=================================="""
embed_criterion = torch.nn.MSELoss().to(opt.device)
adversarial_criterion = torch.nn.BCEWithLogitsLoss().to(opt.device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=opt.lr)
gan_optimizer = torch.optim.Adam(decoder.parameters(), lr=opt.lr)
dis_optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
if opt.lr == 1e-3:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120], gamma=0.1)
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.8*opt.n_epochs)], gamma=0.1)

scaler = GradScaler()
"""===========================TRAINER FUNCTION==============================="""

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_vocab = np.load("/mnt/hdd1/captions/bert_vocab.npy")
bert_model = nn.DataParallel(AutoModel.from_pretrained("bert-base-uncased")).cuda()
bert_model.eval()

adv_weight = 1.0e-4

def train_one_epoch(train_dataloader, model, optimizer, embed_criterion, adversarial_criterion, opt, epoch):
    """
    This function is called every epoch to perform training of the network over one full
    (randomized) iteration of the dataset.
    """
    class_embedding = train_dataloader.dataset.class_embed
    class_names = train_dataloader.dataset.class_name
    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader
    if opt.progressbar:
        data_iterator = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))

    for i, (X, l, Z, _, (image_captions, video_captions)) in enumerate(data_iterator):
        not_broken = l != -1
        X, l, Z = X[not_broken], l[not_broken], Z[not_broken]
        # if i % 20000 == 0:
        #     # Save clip for debugging
        #     clip = X[0].transpose(0, 1).reshape(3, -1, 112, 112)
        #     label = class_names[int(l[0])].replace('/', '_')
        #     batch2gif(clip, int(l[0]), opt.savename + '/samples/samples_train_epoch%d_iter%d_%s' % (epoch, i, label))
        batch_times.append(time.time() - tt_batch)
        # s = list(X.shape)

        X = X.cuda()
        Z = Z.cuda()

        # one_hot = F.one_hot(torch.maximum(image_captions, torch.zeros_like(image_captions)), 3000002).float()
        # image_captions = torch.where((image_captions != -1).unsqueeze(-1), one_hot, torch.zeros_like(one_hot))

        # new_image_captions = list()
        # for image_caption in image_captions:
        #     image_caption = F.one_hot(image_caption, 3000002).float()
        #     if len(image_caption) < 50:
        #         image_caption = F.pad(image_caption, (0, 0, 0, 50 - len(image_caption)))
        #     new_image_captions.append(image_caption)
        # image_captions = torch.stack(new_image_captions, dim=0)
        image_captions = image_captions.cuda()
        video_captions = video_captions.cuda()

        # image_caption_input = dict()
        # for image_caption in image_captions:
        #     for key in image_caption.keys():
        #         if key not in image_caption_input:
        #             image_caption_input[key] = [image_caption[key]]
        #         else:
        #             image_caption_input[key].append(image_caption[key])
        #
        # for key in image_caption_input.keys():
        #     image_caption_input[key] = torch.cat(image_caption_input[key], dim=0).cuda()
        #
        # video_caption_input = dict()
        # for video_caption in video_captions:
        #     for key in video_caption.keys():
        #         if key not in video_caption_input:
        #             video_caption_input[key] = [video_caption[key]]
        #         else:
        #             video_caption_input[key].append(video_caption[key])
        #
        # for key in video_caption_input.keys():
        #     video_caption_input[key] = torch.cat(video_caption_input[key], dim=0).cuda()
        #
        # image_captions = bert_model(**image_captions)
        # video_captions = bert_model(**video_captions)

        # captions = image_captions if random.random() < 0.50 else video_captions
        captions = torch.cat((image_captions, video_captions), dim=1)

        tt_model = time.time()
        with autocast():
            split = 0
            # fake_emb = model(X.to(opt.device))
            # embed_loss = embed_criterion(fake_emb, Z)
            # loss = embed_loss
            split = 0
            # fake_samples, fake_emb, (real_dis, (fake_dis_01, fake_dis_02)) = model(X, captions)
            features = cnn(X)
            fake_samples = decoder(features)
            fake_emb, fake_dis = encoder(fake_samples)

            embed_loss = embed_criterion(fake_emb, Z)
            # g_loss = adversarial_criterion(fake_dis_01 - real_dis.detach(), torch.ones_like(fake_dis_01))
            g_loss = -adversarial_criterion(fake_dis, torch.zeros_like(fake_dis))
            split = 0

        optimizer.zero_grad()
        gan_optimizer.zero_grad()
        scaler.scale(adv_weight * g_loss).backward(retain_graph=True)
        dis_optimizer.zero_grad()
        # optimizer.zero_grad()
        scaler.scale(embed_loss).backward()
        scaler.step(optimizer)
        scaler.step(gan_optimizer)
        # scaler.step(dis_optimizer)

        with autocast():
            # d_loss = adversarial_criterion(real_dis - fake_dis_02, torch.ones_like(real_dis))

            # d_loss_fake = adversarial_criterion(fake_dis_02, torch.zeros_like(fake_dis_02))
            # d_loss_real = adversarial_criterion(real_dis, torch.ones_like(real_dis))
            # d_loss = d_loss_real + d_loss_fake
            #
            # dis_optimizer.zero_grad()
            # scaler.scale(d_loss).backward()
            # scaler.step(dis_optimizer)

            d_loss_sum = 0.0
            d_loss_fake_sum = 0.0
            d_loss_real_sum = 0.0
            for d_i in range(captions.shape[1]):
                _, fake_dis = encoder(fake_samples.detach(), embed=False)
                _, real_dis = encoder(captions[:, d_i], embed=False)

                d_loss_fake = adversarial_criterion(fake_dis, torch.zeros_like(fake_dis))
                d_loss_real = adversarial_criterion(real_dis, torch.ones_like(real_dis))
                d_loss = d_loss_real + d_loss_fake

                scaler.scale(adv_weight * d_loss).backward()

                d_loss_sum += d_loss
                d_loss_fake_sum += d_loss_fake
                d_loss_real_sum += d_loss_real
        scaler.step(dis_optimizer)

        d_loss /= captions.shape[1]
        d_loss_fake /= captions.shape[1]
        d_loss_real /= captions.shape[1]

        adv_loss = g_loss + d_loss
        loss = embed_loss + adv_loss

        # Compute Accuracy.
        pred_embed = fake_emb.detach().cpu().numpy()
        pred_label = cdist(pred_embed, class_embedding, 'cosine').argmin(1)
        acc = accuracy_score(l.numpy(), pred_label) * 100
        accuracy_regressor.append(acc)

        # loss.backward()

        #Update weights using comp. gradients.
        # optimizer.step()

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        #
        # scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        model_times.append(time.time() - tt_model)
        #Store loss per iteration.
        losses.append(loss.item())

        if (epoch * len(data_iterator) + (i + 1)) % 10 == 0:
            txwriter.add_scalar('Train/Loss', loss.item(), epoch * len(data_iterator) + (i + 1))
            txwriter.add_scalar('Train/EmbeddingLoss', embed_loss.item(),epoch * len(data_iterator) + (i + 1))
            txwriter.add_scalar('Train/GeneratorLoss', g_loss.item(), epoch * len(data_iterator) + (i + 1))
            txwriter.add_scalar('Train/DiscriminatorLoss', d_loss.item(), epoch * len(data_iterator) + (i + 1))
            txwriter.add_scalar('Train/DiscriminatorRealLoss', d_loss_real.item(), epoch * len(data_iterator) + (i + 1))
            txwriter.add_scalar('Train/DiscriminatorFakeLoss', d_loss_fake.item(), epoch * len(data_iterator) + (i + 1))
            txwriter.add_scalar('Train/Accuracy', np.mean(acc), epoch * len(data_iterator) + (i + 1))
            split = 0
            # random_index = random.choice(range(len(X)))
            # txwriter.add_text("Train/Caption", " ".join(text_samples[random_index]))
            # videos = ((X.squeeze().detach().cpu().numpy() * 2.0 + 1) * 255.0).astype(np.uint8).permute(0, 2, 1, 3, 4)
            # txwriter.add_video("Train/Video", " ".join(videos[random_index].unsqueeze(0)))
            split = 0
            random_batch_idx = random.choice(range(len(fake_samples)))
            # l, c
            fake_samples = fake_samples.detach().cpu().numpy()[random_batch_idx]
            # l, vocab, c
            distances = np.square(np.expand_dims(bert_vocab, axis=0) - np.expand_dims(fake_samples, axis=1))
            # l, vocab
            distances = np.sum(distances, axis=-1)
            # l
            word_ids = np.argmin(distances, axis=-1)

            sampled_word_ids = word_ids
            decoded_str = tokenizer.decode(sampled_word_ids.tolist())
            # print(decoded_str)
            txwriter.add_text('Train/FakeTextSamples', decoded_str, epoch * len(data_iterator) + (i + 1))


        # if i == len(train_dataloader)-1 or i*opt.bs > 100000:
        #     txwriter.add_scalar('Train/Loss', np.mean(losses), epoch)
        #     txwriter.add_scalar('Train/RegressorAccuracy', np.mean(accuracy_regressor), epoch)
        #     break

        tt_batch = time.time()

    print(Fore.RED, 'Train Accuracy: regressor {0:2.1f}%'.format(np.mean(accuracy_regressor)), Style.RESET_ALL)
    batch_times, model_times = np.sum(batch_times), np.sum(model_times)
    print('TOTAL time for: load the batch %.2f sec, run the model %.2f sec, train %.2f min' % (
                                    batch_times, model_times, (batch_times+model_times)/60))


"""========================================================="""


def evaluate(test_dataloader, txwriter, epoch):
    """
    This function is called every epoch to evaluate the model on 50% of the classes.
    """
    name = test_dataloader.dataset.name
    # _ = model.eval()
    cnn.eval()
    decoder.eval()
    encoder.eval()
    with torch.no_grad():
        ### For all test images, extract features
        n_samples = len(test_dataloader.dataset)

        predicted_embed = np.zeros([n_samples, 300], 'float32')
        true_embed = np.zeros([n_samples, 300], 'float32')
        true_label = np.zeros(n_samples, 'int')
        good_samples = np.zeros(n_samples, 'int') == 1

        final_iter = test_dataloader
        if 'features' not in opt.dataset and opt.progressbar:
            final_iter = tqdm(test_dataloader, desc='Extracting features...')

        fi = 0
        for idx, data in enumerate(final_iter):
            X, l, Z, _ = data
            not_broken = l != -1
            X, l, Z = X[not_broken], l[not_broken], Z[not_broken]
            if len(X) == 0: continue
            # Run network on batch
            # Y = model(X.to(opt.device))
            # _, Y, _ = model(X.to(opt.device))
            features = cnn(X)
            fake_samples = decoder(features)
            Y, _ = encoder(fake_samples)
            Y = Y.cpu().detach().numpy()
            l = l.cpu().detach().numpy()
            predicted_embed[fi:fi + len(l)] = Y
            true_embed[fi:fi + len(l)] = Z.squeeze()
            true_label[fi:fi + len(l)] = l.squeeze()
            good_samples[fi:fi + len(l)] = True
            fi += len(l)

    predicted_embed = predicted_embed[:fi]
    true_embed, true_label = true_embed[:fi], true_label[:fi]

    # Calculate accuracy over test classes
    class_embedding = test_dataloader.dataset.class_embed
    accuracy, accuracy_top5 = compute_accuracy(predicted_embed, class_embedding, true_embed)

    # Logging using tensorboard
    txwriter.add_scalar(name+'/Accuracy', accuracy, epoch)
    txwriter.add_scalar(name+'/Accuracy_Top5', accuracy_top5, epoch)

    # Printing on terminal
    res_str = '%s Epoch %d: Test accuracy: %2.1f%%.' % (name.upper(), epoch, accuracy)
    # res_str = '\n%s Epoch %d: Test accuracy: %2.1f%%, Top5 %2.1f%%.' % (name.upper(), epoch, accuracy, accuracy_top5)

    # Logging accuracy in CSV file
    with open(opt.savename+'/'+name+'_accuracy.csv', 'a') as f:
        f.write('%d, %.1f,%.1f\n' % (epoch, accuracy, accuracy_top5))

    if opt.split == -1:
        # Calculate accuracy per split
        # Only when the model has been trained on a different dataset
        accuracy_split, accuracy_split_top5 = np.zeros(10), np.zeros(10)
        for split in range(len(accuracy_split)):
            # Select test set
            np.random.seed(split) # fix seed for future comparability
            sel_classes = np.random.permutation(len(class_embedding))[:len(class_embedding) // 2]
            sel = [l in sel_classes for l in true_label]
            test_classes = len(sel_classes)

            # Compute accuracy
            subclasses = np.unique(true_label[sel])
            tl = np.array([int(np.where(l == subclasses)[0]) for l in true_label[sel]])
            acc, acc5 = compute_accuracy(predicted_embed[sel], class_embedding[sel_classes], true_embed[sel])
            accuracy_split[split] = acc
            accuracy_split_top5[split] = acc5

        # Printing on terminal
        res_str += ' -- Split accuracy %2.1f%% (+-%.1f) on %d classes' % (
                        accuracy_split.mean(), accuracy_split.std(), test_classes)
        accuracy_split, accuracy_split_std = np.mean(accuracy_split), np.std(accuracy_split)
        accuracy_split_top5, accuracy_split_top5_std = np.mean(accuracy_split_top5), np.std(accuracy_split_top5)

        # Logging using tensorboard
        txwriter.add_scalar(name+'/AccSplit_Mean', accuracy_split, epoch)
        txwriter.add_scalar(name+'/AccSplit_Std', accuracy_split_std, epoch)
        txwriter.add_scalar(name+'/AccSplit_Mean_Top5', accuracy_split_top5, epoch)
        txwriter.add_scalar(name+'/AccSplit_Std_Top5', accuracy_split_top5_std, epoch)

        # Logging accuracy in CSV file
        with open(opt.savename + '/' + name + '_accuracy_splits.csv', 'a') as f:
            f.write('%d, %.1f,%.1f,%.1f,%.1f\n' % (epoch, accuracy_split, accuracy_split_std,
                                                   accuracy_split_top5, accuracy_split_top5_std))
    print(Fore.GREEN, res_str, Style.RESET_ALL)
    return accuracy, accuracy_top5


def compute_accuracy(predicted_embed, class_embed, true_embed):
    """
    Compute accuracy based on the closest Word2Vec class
    """
    assert len(predicted_embed) == len(true_embed), "True and predicted labels must have the same number of samples"
    y_pred = cdist(predicted_embed, class_embed, 'cosine').argsort(1)
    y = cdist(true_embed, class_embed, 'cosine').argmin(1)
    accuracy = accuracy_score(y, y_pred[:, 0]) * 100
    accuracy_top5 = np.mean([l in p for l, p in zip(y, y_pred[:, :5])]) * 100
    return accuracy, accuracy_top5


"""===================SCRIPT MAIN========================="""

if __name__ == '__main__':
    trainsamples = 0
    if not opt.evaluate:
        trainsamples = len(dataloaders['training'][0].dataset)
        with open(opt.savename + '/train_samples_%d_%d.txt' % (opt.n_classes, trainsamples), 'w') as f:
            f.write('%d, %d\n' % (opt.n_classes, trainsamples) )

    best_acc = 0
    print('\n----------')
    txwriter = SummaryWriter(logdir=opt.savename)
    epoch_times = []
    for epoch in range(opt.n_epochs):
        print('\n{} classes {} from {}, LR {} BS {} CLIP_LEN {} N_CLIPS {} OVERLAP {} SAMPLES {}'.format(
                    opt.network.upper(), opt.n_classes,
                    opt.dataset.upper(), opt.lr, opt.bs, opt.clip_len, opt.n_clips,
                    opt.class_overlap, trainsamples))
        print(opt.savename)
        tt = time.time()

        ## Train one epoch
        if not opt.evaluate:
            # _ = model.train()
            cnn.train()
            decoder.train()
            encoder.train()
            train_one_epoch(dataloaders['training'][0], cnn, optimizer,
                            embed_criterion, adversarial_criterion, opt, epoch)

        ### Evaluation
        accuracies = []
        for test_dataloader in dataloaders['testing']:
            accuracy, _ = evaluate(test_dataloader, txwriter, epoch)
            accuracies.append(accuracy)
        accuracy = np.mean(accuracies)

        if accuracy > best_acc:
            # Save best model
            # torch.save({'state_dict': model.state_dict(), 'opt': opt, 'accuracy': accuracy},
            #            opt.savename + '/checkpoint.pth.tar')
            best_acc = accuracy

        #Update the Metric Plot and save it.
        epoch_times.append(time.time() - tt)
        print('----- Epoch ', Fore.RED, '%d' % epoch, Style.RESET_ALL,
              'done in %.2f minutes. Remaining %.2f minutes.' % (
              epoch_times[-1]/60, ((opt.n_epochs-epoch-1)*np.mean(epoch_times))/60),
              Fore.BLUE, 'Best accuracy %.1f' % best_acc, Style.RESET_ALL)
        # scheduler.step(accuracy)
        scheduler.step()
        opt.lr = optimizer.param_groups[0]['lr']

        if opt.evaluate:
            break

    txwriter.close()

