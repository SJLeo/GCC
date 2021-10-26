import argparse
import os
import shutil
from PIL import Image
import zipfile
import tarfile
import urllib.request
from tqdm import tqdm

parser = argparse.ArgumentParser('pretrain_sr_datasets')
parser.add_argument('--output_path', type=str, default='database/sr', help='the path of output dataroot')
parser.add_argument('--min_image_size', type=int, default=96)
args = parser.parse_args()

def unzip_zip_file(zip_path, data_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()

def unzip_tar_file(zip_path, data_path):
    tar_ref = tarfile.open(zip_path, "r:")
    tar_ref.extractall(data_path)
    tar_ref.close()

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    print("[!] download data file")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_dataset(output_path, min_image_size):
    COCO_train_URL = "http://images.cocodataset.org/zips/train2014.zip"
    COCO_val_URL = "http://images.cocodataset.org/zips/val2014.zip"
    Set5_Set14_BSD100_Urban100_URL = "https://cv.snu.ac.kr/research/EDSR/benchmark.tar"

    if not os.path.exists('temp'):
        os.makedirs('temp')

    train_hr = os.path.join(output_path, 'train')
    set5_hr = os.path.join(output_path, 'test/Set5')
    set14_hr = os.path.join(output_path, 'test/Set14')
    b100_hr = os.path.join(output_path, 'test/B100')
    urban100_hr = os.path.join(output_path, 'test/Urban100')
    if not os.path.exists(train_hr):
        os.makedirs(train_hr)
    if not os.path.exists(set5_hr):
        os.makedirs(set5_hr)
    if not os.path.exists(set14_hr):
        os.makedirs(set14_hr)
    if not os.path.exists(b100_hr):
        os.makedirs(b100_hr)
    if not os.path.exists(urban100_hr):
        os.makedirs(urban100_hr)

    # download_url(COCO_train_URL, os.path.join('temp', 'train2014.zip'))
    # download_url(COCO_val_URL, os.path.join('temp', 'val2014.zip'))
    # download_url(Set5_Set14_BSD100_Urban100_URL, os.path.join('temp', 'benchmark.tar'))

    print('[!] Upzip zipfile')
    unzip_zip_file(os.path.join('temp', 'train2014.zip'), 'temp')
    unzip_zip_file(os.path.join('temp', 'val2014.zip'), 'temp')
    unzip_tar_file(os.path.join('temp', 'benchmark.tar'), 'temp')

    print('[!] Reformat COCO train HR')
    hrimage_list = sorted(os.listdir('temp/train2014/'))
    for index, img_name in enumerate(hrimage_list):
        path = os.path.join('temp/train2014/', img_name)
        img = Image.open(path, mode='r').convert('RGB')

        if img.width >= min_image_size and img.height >= min_image_size:
            shutil.move(path, os.path.join(train_hr, img_name))

    print('[!] Reformat COCO val HR')
    hrimage_list = sorted(os.listdir('temp/val2014/'))
    for index, img_name in enumerate(hrimage_list):
        path = os.path.join('temp/val2014/', img_name)
        img = Image.open(path, mode='r').convert('RGB')

        if img.width >= min_image_size and img.height >= min_image_size:
            shutil.move(path, os.path.join(train_hr, img_name))

    print('[!] Reformat Set5 HR')
    hrimage_list = sorted(os.listdir('temp/benchmark/Set5/HR/'))
    for index, img_name in enumerate(hrimage_list):
        path = os.path.join('temp/benchmark/Set5/HR/', img_name)
        shutil.move(path, os.path.join(set5_hr, img_name))

    print('[!] Reformat Set14 HR')
    hrimage_list = sorted(os.listdir('temp/benchmark/Set14/HR/'))
    for index, img_name in enumerate(hrimage_list):
        path = os.path.join('temp/benchmark/Set14/HR/', img_name)
        shutil.move(path, os.path.join(set14_hr, img_name))

    print('[!] Reformat BSD100 HR')
    hrimage_list = sorted(os.listdir('temp/benchmark/B100/HR/'))
    for index, img_name in enumerate(hrimage_list):
        path = os.path.join('temp/benchmark/B100/HR/', img_name)
        shutil.move(path, os.path.join(b100_hr, img_name))

    print('[!] Reformat Urban100 HR')
    hrimage_list = sorted(os.listdir('temp/benchmark/Urban100/HR/'))
    for index, img_name in enumerate(hrimage_list):
        path = os.path.join('temp/benchmark/Urban100/HR/', img_name)
        shutil.move(path, os.path.join(urban100_hr, img_name))

    # shutil.rmtree('temp')

if __name__ == '__main__':
    download_dataset(args.output_path, args.min_image_size)