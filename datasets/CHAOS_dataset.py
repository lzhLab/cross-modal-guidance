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
import pickle

class chaosDataset(Dataset):

    def __init__(self, paths, transforms=None):
        super().__init__()
        self.paths = paths
        self.transforms = transforms

    def __getitem__(self, item):
        t1_paths = self.paths[item]
        t1_dir = r'/T1'
        t2_dir = r'/T2'
        # t1_dir = r'/inphase'
        # t2_dir = r'/outphase'
        train_dir = r'/image'
        mask_dir = r'/gt'

        if isinstance(t1_paths, list):
            t1_imgs = []
            t2_imgs = []
            for t1_path in t1_paths:
                t2_path = t1_path.replace(t1_dir, t2_dir)
                t1_imgs.append(readdat(t1_path))
                t2_imgs.append(readdat(t2_path))
            t1_mask_path = t1_paths[len(t1_paths) // 2].replace(train_dir, mask_dir).replace('.dat', '.png')
            t2_mask_path = t1_mask_path.replace(t1_dir, t2_dir).replace('.dat', '.png')

        else:
            t2_paths = t1_paths.replace(t1_dir, t2_dir)
            t1_imgs = readdat(t1_paths)
            t2_imgs = readdat(t2_paths)

            t1_mask_path = t1_paths.replace(train_dir, mask_dir).replace('.dat', '.png')
            t2_mask_path = t1_mask_path.replace(t1_dir, t2_dir).replace('.dat', '.png')

        t1_mask = Image.open(t1_mask_path).convert('L')
        t2_mask = Image.open(t2_mask_path).convert('L')

        if self.transforms is not None:
            t1_imgs, t2_imgs, t1_mask, t2_mask = self.transforms(t1_imgs, t2_imgs, t1_mask, t2_mask)

        # if isinstance(t1_paths, list):
        #     t1_imgs, t2_imgs= torch.cat(t1_imgs), torch.cat(t2_imgs)
        
        return t1_imgs, t2_imgs, t1_mask, t2_mask, torch.tensor([])

    def __len__(self):
        return len(self.paths)

def readdat(path):
    with open(path, 'rb') as dat:
        img = pickle.load(dat)
        img = (img - img.min()) / (img.max() - img.min())
    return Image.fromarray(img)


class VolumnCHAOSDataset(Dataset):

    def __init__(self, paths, transforms=None, train=True, size=None):
        super().__init__()
        self.paths = paths
        self.transforms = transforms
        self.train = train
        self.size = size # (16, 96, 96)

    def __getitem__(self, item):

        t1_paths, spacing = self.paths[item]

        t1_imgs = []
        t2_imgs = []
        t1_masks = []
        t2_masks = []
        
        for t1_path in t1_paths:

            t1_img, t2_img, t1_mask, t2_mask = self._get_data_from_path(t1_path)

            t1_imgs.append(t1_img)
            t2_imgs.append(t2_img)
            t1_masks.append(t1_mask)
            t2_masks.append(t2_mask)

        if self.transforms is not None:
            t1_imgs, t2_imgs, t1_masks, t2_masks = self.transforms(t1_imgs, t2_imgs, t1_masks, t2_masks)

        if self.train:
            t1_imgs, t2_imgs, t1_masks, t2_masks = self.randomCrop(t1_imgs, t2_imgs, t1_masks, t2_masks)

        # return t1_imgs_1.unsqueeze(0), t2_imgs_1.unsqueeze(0), t1_masks_1.unsqueeze(0), t2_masks_1.unsqueeze(0), torch.tensor(spacing), t1_imgs_2.unsqueeze(0), t2_imgs_2.unsqueeze(0)
        return t1_imgs.unsqueeze(0), t2_imgs.unsqueeze(0), t1_masks.unsqueeze(0), t2_masks.unsqueeze(0), torch.tensor(spacing)

    def __len__(self):
        return len(self.paths)
    
    def readdat(self, path):
        with open(path, 'rb') as dat:
            img = pickle.load(dat)
            img = (img - img.min()) / (img.max() - img.min())
        return img
    
    def _get_data_from_path(self, t1_path):
        t1_dir = r'/T1'
        t2_dir = r'/T2'
        # t1_dir = r'/inphase'
        # t2_dir = r'/outphase'
        train_dir = r'/image'
        mask_dir = r'/gt'

        t2_path = t1_path.replace(t1_dir, t2_dir)
        t1_mask_path = t1_path.replace(train_dir, mask_dir).replace('.dat', '.png')
        t2_mask_path = t1_mask_path.replace(t1_dir, t2_dir)

        t1_img = self.readdat(t1_path)
        t2_img = self.readdat(t2_path)
        t1_mask = Image.open(t1_mask_path).convert('L')
        t2_mask = Image.open(t2_mask_path).convert('L')

        return Image.fromarray(t1_img), Image.fromarray(t2_img), t1_mask, t2_mask

    def randomCrop(self, t1_imgs, t2_imgs, t1_masks, t2_masks):
        # size:(16,96,96)
        size_Z, size_H, size_W = self.size
        Z, H, W = t1_imgs.size()
        point_Z = random.randint(0, Z - size_Z)
        point_H = random.randint(0, H - size_H)
        point_W = random.randint(0, W - size_W)
        # print(point_Z, point_H, point_W)
        t1_imgs_crop = t1_imgs[point_Z:point_Z + size_Z, point_H:point_H + size_H, point_W:point_W + size_W]
        t2_imgs_crop = t2_imgs[point_Z:point_Z + size_Z, point_H:point_H + size_H, point_W:point_W + size_W]
        t1_masks_crop = t1_masks[point_Z:point_Z + size_Z, point_H:point_H + size_H, point_W:point_W + size_W]
        t2_masks_crop = t2_masks[point_Z:point_Z + size_Z, point_H:point_H + size_H, point_W:point_W + size_W]

        return t1_imgs_crop, t2_imgs_crop, t1_masks_crop, t2_masks_crop

def split_data(folderpath, nfolds=5, expand_size=0, random_state=1):
    '''
    :param folderpath:
    :param nfolds:
    :param expand_size:
    :param random_state:
    :return: train_paths, val_paths
    '''

    spacing_path = r'/share/users_root/masters/ywm/multimodal-seg-net/datasets/data/chaos_spacing.dat'
    spacing_file = open(spacing_path, 'rb')
    spacing_dict = pickle.load(spacing_file)
    spacing_file.close()

    #
    group = []
    fold_names = os.listdir(folderpath)
    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    for train_idx, val_idx in kf.split(fold_names):
        trainpart = [fold_names[i] for i in train_idx]
        valpart = [fold_names[i] for i in val_idx]
        print(trainpart, valpart)
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
            data = (get_3d_data(folderpath, foldername, expand_size), spacing_dict[foldername])
            val_paths.append(data)
        result.append((train_paths, val_paths))

    return result


def expand_data(foldername, expand_size=0):
    '''
    获取文件夹中的图片文件名，返回一个list
    foldername: 文件夹名称
    expand_size: 要扩展的图片张数
    '''
    group = []
    for name in os.listdir(foldername):
        if name.endswith('dat'):
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


def split_data_folder(folderpath, expand_size=0, nfolds=5, random_state=1):

    '''
    :param folderpath:
    :param nfolds:
    :param expand_size:
    :param random_state:
    :return: train_paths, val_paths
    '''

    spacing_path = r'/share/users_root/masters/ywm/multimodal-seg-net/datasets/data/chaos_spacing.dat'
    spacing_file = open(spacing_path, 'rb')
    spacing_dict = pickle.load(spacing_file)
    spacing_file.close()

    group = []
    fold_names = os.listdir(folderpath)
    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    for train_idx, val_idx in kf.split(fold_names):
        
        # trainpart = [fold_names[i] for i in train_idx]
        # valpart = [fold_names[i] for i in val_idx]
        trainpart = [(get_3d_data(folderpath, fold_names[i], expand_size), spacing_dict[fold_names[i]]) for i in train_idx]
        valpart = [(get_3d_data(folderpath, fold_names[i], expand_size), spacing_dict[fold_names[i]]) for i in val_idx]
        # trainpart = [get_3d_data(folderpath, fold_names[i], expand_size) for i in train_idx]
        # valpart = [get_3d_data(folderpath, fold_names[i], expand_size) for i in val_idx]
        group.append([trainpart, valpart])

    return group


def get_3d_data(folderpath, fold_name, expand_size=0):

    folder_name = os.path.join(folderpath, fold_name)
    png_name = []
    png_list = os.listdir(folder_name)
    for file in png_list:
        png_name.append(file.split('.')[-2])
    png_name.sort(key=lambda str: int(str))
    result = [os.path.join(folder_name, name + '.dat') for name in png_name]

    # if expand_size > 0:
    #     expand_result = []
    #     for i in range(expand_size, len(result) - expand_size):
    #         expand_result.append(result[i - expand_size: i + expand_size + 1])
    #     return expand_result

    return result

if __name__ == '__main__':

    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # random.seed(1)
    # np.random.seed(1)
    # torch.manual_seed(1)
    # os.environ['PYTHONHASHSEED'] = str(1)
    # if torch.cuda.is_available():
    #     # torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(1)
    #     # cudnn.deterministic = True
    #     # cudnn.benchmark = False
    
    # # 五折划分数据集
    # data_path = r'/home/student/ywm/multimodal-seg-net/datasets/data/CHAOS/T1/image'
    # data = split_data_folder(data_path, expand_size=0)

    # print(data)

    trainTransform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation([-15, 15]),
        CenterCrop(256),
        ToTensor(),
        # Normalize([0.], [1.])
    ])

    path = r'/share/users_root/masters/ywm/multimodal-seg-net/datasets/data/CHAOS_T1_in_out/inphase/image'
    data =split_data(path, nfolds=5, expand_size=1, random_state=1)
    for train_paths, val_paths in data:
        print(train_paths)

    # for train_path, val_path in data:
        # print(len(val_path))



    # trainset = VolumnCHAOSDataset(train_paths, transforms=trainTransform, train=True, size=(16,96,96))
    # train_dataloader = DataLoader(trainset, batch_size=2, shuffle=False, num_workers=8)
    # for t1_imgs, t2_imgs, t1_mask, t2_mask in train_dataloader:
        # pass
        # temp=0
        # print((t1_mask == 1.).float().sum())
        # t1_imgs = t1_imgs.squeeze().numpy()
        # # t1_imgs = (t1_imgs - t1_imgs.min()) / (t1_imgs.max() - t1_imgs.min())
        # cv2.imwrite('0.png', t1_imgs * 255.)
        # print(len(t1_imgs.size()))
        # print(t1_mask.size())

