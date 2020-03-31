import os
import json
import errno
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch import nn
import torchvision.transforms as transforms


# ################################# 9.2.1 Pikachu dataset ################################
# pikachu数据集下载相关

def gen_bar_updater():
    pbar = tqdm(total=None)
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)
    return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


# 辅助函数：下载指定url
def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e

# The following code is used to download pikachu dataset from mxnet aws server in the form of .rec files
# Then this .rec file is converted into png images and json files for annotation data
# This part requires 'mxnet' library which can be downloaded using conda 
# using the command 'conda install mxnet'
# Matplotlib is also required for saving the png image files

# Download Pikachu Dataset
def download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        download_url(root_url + k, data_dir)


# Create dataloaders in mxnet
def load_data_pikachu_rec_mxnet(batch_size, edge_size=256, data_dir='../../../dataset/pikachu'):
    from mxnet import image
    """Load the pikachu dataset"""
    download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # The shape of the output image
        # shuffle=True,  # Read the data set in random order
        # rand_crop=1,  # The probability of random cropping is 1
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), shuffle=False)
    return train_iter, val_iter


# Use mxnet dataloaders to convert .rec file to .png images and annotations.json
def download_and_preprocess_pikachu_data(dir='../../../dataset/pikachu/'):

    if os.path.exists(os.path.join(dir, 'train')) and os.path.exists(os.path.join(dir, 'val')):
        return

    import mxnet as mx
    batch_size = 1
    ctx = mx.cpu(0)

    train_iter, val_iter = load_data_pikachu_rec_mxnet(batch_size, edge_size=256, data_dir=dir)

    os.mkdir(os.path.join(dir, 'train'))
    os.mkdir(os.path.join(dir, 'val'))
    os.mkdir(os.path.join(dir, 'train/images'))
    os.mkdir(os.path.join(dir, 'val/images'))

    annotations_train = dict()
    train_iter.reset()  # Read data from the start.
    id = 0
    for batch in train_iter:
        id+=1

        X = batch.data[0].as_in_context(ctx)

        Y = batch.label[0].as_in_context(ctx)

        x = X.asnumpy()
        x = x.transpose((2,3,1,0))
        x = x.squeeze(axis=-1)
        plt.imsave(os.path.join(dir, 'train/images', 'pikachu_' + str(id) + '.png'), x/255.)
        an = dict()
        y = Y.asnumpy()

        an['class'] = y[0, 0][0].tolist()
        an['loc'] = y[0,0][1:].tolist()
        an['id'] = [id]
        an['image'] = 'pikachu_' + str(id) + '.png'
        annotations_train['data_' + str(id)] = an

    import json
    with open(os.path.join(dir, 'train', 'annotations.json'), 'w') as outfile:
        json.dump(annotations_train, outfile)
    outfile.close()


    annotations_val = dict()
    val_iter.reset()  # Read data from the start.
    id = 0
    for batch in val_iter:
        id+=1

        X = batch.data[0].as_in_context(ctx)

        Y = batch.label[0].as_in_context(ctx)

        x = X.asnumpy()
        x = x.transpose((2,3,1,0))
        x = x.squeeze(axis=-1)
        plt.imsave(os.path.join(dir, 'val/images', 'pikachu_' + str(id) + '.png'), x/255.)
        an = dict()
        y = Y.asnumpy()

        an['class'] = y[0, 0][0].tolist()
        an['loc'] = y[0,0][1:].tolist()
        an['id'] = [id]
        an['image'] = 'pikachu_' + str(id) + '.png'
        annotations_val['data_' + str(id)] = an

    import json
    with open(os.path.join(dir, 'val', 'annotations.json'), 'w') as outfile:
        json.dump(annotations_val, outfile)
    outfile.close()


# Create Dataloaders in Pytorch 
class PIKACHU(torch.utils.data.Dataset):
    def __init__(self, data_dir, set, transform=None, target_transform=None):

        self.image_size = (3, 256, 256)
        self.images_dir = os.path.join(data_dir, set, 'images')

        self.set = set
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.target_transform = target_transform

        annotations_file = os.path.join(data_dir, set, 'annotations.json')
        with open(annotations_file) as file:
            self.annotations = json.load(file)

    def __getitem__(self, index):

        annotations_i = self.annotations['data_' + str(index+1)]

        image_path = os.path.join(self.images_dir, annotations_i['image'])
        img = np.array(Image.open(image_path).convert('RGB').resize((self.image_size[2], self.image_size[1]), Image.BILINEAR))
        # print(img.shape)
        loc = np.array(annotations_i['loc'])

        label = 1 - annotations_i['class']

        if self.transform is not None:
            img = self.transform(img)
        return (img, loc, label)

    def __len__(self):
        return len(self.annotations)
