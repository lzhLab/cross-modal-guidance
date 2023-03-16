import argparse
import sys
import os, cv2
sys.path.append("..")
import numpy as np
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from utils.transforms import *
import glob
import pickle


class VolumnlabeledDataset(Dataset):

    def __init__(self, paths, transforms=None):
        super().__init__()
        self.paths = paths
        self.transforms = transforms
        
        self.pv_dir = r'/jing'
        self.art_dir = r'/registered_dong'
        self.train_dir = r'/train_img'
        self.mask_dir = r'/tumor_mask'
        self.liver_dir = r'/liver_mask'

        print()

    def __getitem__(self, item):

        pv_paths, spacing = self.paths[item]

        pv_imgs = []
        art_imgs = []
        pv_masks = []
        art_masks = []
        
        dir_name = None

        ii = False
        for pv_path in pv_paths:
            if isinstance(pv_path, list):
                pv_img_temp = []
                art_img_temp = []
                pv_mask_temp = []
                art_mask_temp = []
                for i in pv_path:
                    pv_img, art_img, pv_mask, art_mask = self._get_data_from_path(i)
                    pv_img_temp.append(pv_img)
                    art_img_temp.append(art_img)
                    pv_mask_temp.append(pv_mask)
                    art_mask_temp.append(art_mask)
                
                if self.transforms is not None:
                    pv_img_temp, art_img_temp, pv_mask_temp, art_mask_temp = self.transforms(pv_img_temp, art_img_temp, pv_mask_temp, art_mask_temp)


                pv_img_temp, art_img_temp, pv_mask_temp, art_mask_temp = torch.cat(tuple(pv_img_temp)), torch.cat(tuple(art_img_temp)), torch.cat(tuple(pv_mask_temp)), torch.cat(tuple(art_mask_temp))
                
                pv_mask_temp = pv_mask_temp[len(pv_mask_temp) // 2]
                art_mask_temp = art_mask_temp[len(art_mask_temp) // 2]
                pv_imgs.append(pv_img_temp.unsqueeze(0))
                art_imgs.append(art_img_temp.unsqueeze(0))
                pv_masks.append(pv_mask_temp.unsqueeze(0))
                art_masks.append(art_mask_temp.unsqueeze(0))
            else:
                ii = True
                pv_img, art_img, pv_mask, art_mask = self._get_data_from_path(pv_path)
                pv_imgs.append(pv_img)
                art_imgs.append(art_img)
                pv_masks.append(pv_mask)
                art_masks.append(art_mask)

        if self.transforms is not None and ii:
            pv_imgs, art_imgs, pv_masks, art_masks = self.transforms(pv_imgs, art_imgs, pv_masks, art_masks)
            
            pv_imgs = pv_imgs[1:-1]
            art_imgs = art_imgs[1:-1]
            pv_masks = pv_masks[1:-1]
            art_masks = art_masks[1:-1]

        if isinstance(pv_paths, list):
            pv_imgs, art_imgs, pv_masks, art_masks = torch.cat(tuple(pv_imgs)), torch.cat(tuple(art_imgs)), torch.cat(tuple(pv_masks)), torch.cat(tuple(art_masks))

        return pv_imgs, art_imgs, pv_masks, art_masks, torch.tensor(spacing)

    def __len__(self):
        return len(self.paths)
    
    def _get_data_from_path(self, pv_path):
        pv_dir = r'/jing'
        art_dir = r'/registered_dong'
        train_dir = r'/train_img'
        mask_dir = r'/tumor_mask'
        liver_dir = r'/liver_mask'
        art_path = pv_path.replace(pv_dir, art_dir)
        pv_liver_path = pv_path.replace(train_dir, liver_dir)
        art_liver_path = pv_liver_path.replace(pv_dir, art_dir)
        pv_liver_mask = Image.open(pv_liver_path).convert('L')
        art_liver_mask = Image.open(art_liver_path).convert('L')
        pv_img = masking(Image.fromarray(torch.load(pv_path)), pv_liver_mask)
        art_img = masking(Image.fromarray(torch.load(art_path)), art_liver_mask)

        pv_mask_path = pv_path.replace(train_dir, mask_dir)
        art_mask_path = pv_mask_path.replace(pv_dir, art_dir)
        pv_mask = Image.open(pv_mask_path).convert('L')
        art_mask = Image.open(art_mask_path).convert('L')

        return pv_img, art_img, pv_mask, art_mask

def masking(pv_img, liver_mask):
    pv_img_np = np.array(pv_img)
    liver_mask_np = np.array(liver_mask)
    return Image.fromarray(pv_img_np * liver_mask_np)


def split_data_folder(folderpath, expand_size=0, nfolds=5, random_state=1):

    '''
    :param folderpath:
    :param nfolds:
    :param expand_size:
    :param random_state:
    :return: train_paths, val_paths
    '''

    spacing_path = r'/share/users_root/masters/ywm/multimodal-seg-net/datasets/data/jing_spacing.dat'
    spacing_file = open(spacing_path, 'rb')
    spacing_dict = pickle.load(spacing_file)
    spacing_file.close()

    group = []
    fold_names = os.listdir(folderpath)
    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    for train_idx, val_idx in kf.split(fold_names):
        trainpart = [(get_3d_data(folderpath, fold_names[i], expand_size), spacing_dict[fold_names[i]]) for i in train_idx]
        valpart = [(get_3d_data(folderpath, fold_names[i], expand_size), spacing_dict[fold_names[i]]) for i in val_idx]
        group.append([trainpart, valpart])

    return group


def get_3d_data(folderpath, fold_name, expand_size=0):

    folder_name = os.path.join(folderpath, fold_name)
    png_name = []
    png_list = os.listdir(folder_name)
    for file in png_list:
        png_name.append(file.split('.')[-2])
    png_name.sort(key=lambda str: int(str))
    result = [os.path.join(folder_name, name + '.png') for name in png_name]

    if expand_size > 0:
        expand_result = []
        for i in range(expand_size, len(result) - expand_size):
            expand_result.append(result[i - expand_size: i + expand_size + 1])
        return expand_result

    return result



if __name__ == '__main__':

    data_path = r'/share/users_root/masters/ywm/multimodal-seg-net/datasets/data/labeled_refined/train_img/jing'
    group = split_data_folder(data_path, expand_size=1)
    trainpath, valpath = group[0]

    # for i in valpath:
    #     png_list, spacing = valpath[0]
    #     print(png_list)
    #     print(spacing)
    #     break

    valTransform = Compose_Test([ToTensor_Test()])

    valset = VolumnlabeledDataset(valpath, valTransform)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    # print(len(valloader.dataset))

    for pv_imgs, art_imgs, pv_mask, art_mask, spacing in valloader:
        # for i in range(pv_imgs.size()[1]):
        #     if (len(pv_imgs[:, i, :, :].squeeze().size()) == 2):
        #         print(True)
            print(pv_imgs.size())
            print(spacing)
            # print()