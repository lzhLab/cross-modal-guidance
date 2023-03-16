import sys
import os, cv2
sys.path.append("..")
import numpy as np
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from utils.transforms import *
import glob
import random

class labeledDataset(Dataset):

    def __init__(self, paths, transforms=None, train=False):
        super(labeledDataset, self).__init__()
        self.paths = paths
        self.transforms = transforms
        self.training = train

    def __getitem__(self, item):
        pv_paths = self.paths[item]
        pv_dir = r'/pv'
        art_dir = r'/art'
        train_dir = r'/train_img'
        mask_dir = r'/tumor_mask'
        liver_dir = r'/liver_mask'

        if isinstance(pv_paths, list):
            pv_imgs = []
            art_imgs = []
            for pv_path in pv_paths:
                art_path = pv_path.replace(pv_dir, art_dir)
                pv_liver_path = pv_path.replace(train_dir, liver_dir)
                art_liver_path = pv_liver_path.replace(pv_dir, art_dir)
                pv_liver_mask = Image.open(pv_liver_path).convert('L')
                art_liver_mask = Image.open(art_liver_path).convert('L')
                pv_imgs.append(masking(Image.fromarray(torch.load(pv_path)), pv_liver_mask))
                art_imgs.append(masking(Image.fromarray(torch.load(art_path)), art_liver_mask))
            pv_mask_path = pv_paths[len(pv_paths) // 2].replace(train_dir, mask_dir)
            art_mask_path = pv_mask_path.replace(pv_dir, art_dir)
            mid_liver_path = pv_paths[len(pv_paths) // 2].replace(train_dir, liver_dir)

        else:
            art_path = pv_paths.replace(pv_dir, art_dir)
            pv_liver_path = pv_paths.replace(train_dir, liver_dir)
            art_liver_path = pv_liver_path.replace(pv_dir, art_dir)

            pv_liver_mask = Image.open(pv_liver_path).convert('L')
            art_liver_mask = Image.open(art_liver_path).convert('L')
            pv_imgs = masking(Image.fromarray(torch.load(pv_paths)), pv_liver_mask)
            art_imgs = masking(Image.fromarray(torch.load(art_path)), art_liver_mask)

            pv_mask_path = pv_paths.replace(train_dir, mask_dir)
            art_mask_path = pv_mask_path.replace(pv_dir, art_dir)
            mid_liver_path = pv_liver_path

        pv_mask = Image.open(pv_mask_path).convert('L')
        art_mask = Image.open(art_mask_path).convert('L')

        # if self.training:
        #     pv_imgs, art_imgs, pv_mask, art_mask = self.randomCrop(pv_imgs, art_imgs, pv_mask, art_mask, mid_liver_path, 224)

        if self.transforms is not None:
            pv_imgs, art_imgs, pv_mask, art_mask = self.transforms(pv_imgs, art_imgs, pv_mask, art_mask)
        

        return pv_imgs, art_imgs, pv_mask, art_mask

    def __len__(self):
        return len(self.paths)

    def randomCrop(self, pv_imgs, art_imgs, pv_mask, art_mask, mid_liver_path, res):
        liver = cv2.imread(mid_liver_path, cv2.IMREAD_GRAYSCALE)
        liver_contours, hierarchy = cv2.findContours(liver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minX = 512
        minY = 512
        maxX = 0
        maxY = 0
        for i in range(len(liver_contours)):
            contour = liver_contours[i]
            temp_minX = np.min(contour[:,:,1])
            temp_minY = np.min(contour[:,:,0])
            temp_maxX = np.max(contour[:,:,1])
            temp_maxY = np.max(contour[:,:,0])
            if minX > temp_minX:
                minX = temp_minX
            if minY > temp_minY:
                minY = temp_minY
            if maxX < temp_maxX:
                maxX = temp_maxX
            if maxY < temp_maxY:
                maxY = temp_maxY
        # print(minX, minY, maxX, maxY, 1)
        stx = random.randint(minX, max(minX, maxX - res))
        sty = random.randint(minY, max(minY, maxY - res))
        if stx + res > 512:
            stx = 512 - res
        if sty + res > 512:
            sty = 512 - res
        pv_imgs = crop(pv_imgs, stx, sty, res)
        art_imgs = crop(art_imgs, stx, sty, res)
        pv_mask = crop(pv_mask, stx, sty, res)
        art_mask = crop(art_mask, stx, sty, res)
        return pv_imgs, art_imgs, pv_mask, art_mask


def crop(img, stx, sty, res):
    if isinstance(img, list):
        result = []
        for i in img:
            i = np.array(i)
            i = i[stx:stx + res, sty:sty + res]
            result.append(Image.fromarray(i))
    else:
        i = np.array(img)
        i = i[stx:stx + res, sty:sty + res]
        result = Image.fromarray(i)
    return result




def masking(pv_img, liver_mask):
    pv_img_np = np.array(pv_img)
    liver_mask_np = np.array(liver_mask)
    return Image.fromarray(pv_img_np * liver_mask_np)


def split_data(folderpath, nfolds=5, expand_size=0, random_state=1):

    group = []
    fold_names = os.listdir(folderpath)
    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    for train_idx, val_idx in kf.split(fold_names):
        trainpart = [fold_names[i] for i in train_idx]
        valpart = [fold_names[i] for i in val_idx]
        group.append([trainpart, valpart])

    #
    result = []
    for fold in range(nfolds):

        train_paths = []
        val_paths = []
        train_folders = group[fold][0]
        val_folders = group[fold][1]

        for foldername in train_folders:
            train_folder = os.path.join(folderpath, foldername)
            expand_list = expand_data(train_folder, expand_size=expand_size)
            for data in expand_list:
                train_paths.append(data)
        for foldername in val_folders:
            val_folder = os.path.join(folderpath, foldername)
            expand_list = expand_data(val_folder, expand_size=expand_size)
            for data in expand_list:
                val_paths.append(data)
        result.append((train_paths, val_paths))

    return result


def expand_data(foldername, expand_size=0):

    group = []
    for name in os.listdir(foldername):
        if name.endswith('png'):
            group.append(os.path.join(foldername, name))
    if expand_size == 0:   #若连续张数为0，直接返回group
        return group
    else:                   #否则获取连续的图片list
        dict = {}
        result = []
        for filename in group:
            num = filename.split('/')[-1].split('.')[0]
            dict[int(num)] = filename

        for filename in group:
            sub_group = []
            for i in range(0 - expand_size, 1 + expand_size):
                num = filename.split('/')[-1].split('.')[0]
                if dict.get(int(num) + i):
                    sub_group.append(dict[int(num) + i])
                else:
                    break
                if i == expand_size:
                    result.append(sub_group)
    return result


# if __name__ == '__main__':
#
#     trainTransform = Compose([
#         RandomHorizontalFlip(p=0.5),
#         RandomVerticalFlip(p=0.5),
#         ToTensor_LITS()
#     ])
#
#     path = r''
#     data =split_data(path, nfolds=5, expand_size=0, random_state=1)
#     train_paths, val_paths = data[0]
#     trainset = labeledDataset(train_paths, transforms=trainTransform)
#     train_dataloader = DataLoader(trainset, batch_size=12, shuffle=True, num_workers=4)
#     for pv_imgs, art_imgs, pv_mask, art_mask in train_dataloader:
#
#         # temp=0
#         print(type(pv_imgs))
#         print(pv_imgs.dtype)
#         print(torch.max(pv_imgs))
#         print(torch.min(pv_imgs))
