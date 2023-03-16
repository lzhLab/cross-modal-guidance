import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np
import random

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):
        for t in self.transforms:
            pv_imgs, art_imgs, pv_mask, art_mask = t(pv_imgs, art_imgs, pv_mask, art_mask)
        return pv_imgs, art_imgs, pv_mask, art_mask


class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):

        if torch.rand(1) < self.p:
            if isinstance(pv_imgs, list):
                for i in range(len(pv_imgs)):
                    pv_imgs[i] = F.hflip(pv_imgs[i])
                    art_imgs[i] = F.hflip(art_imgs[i])
            else:
                pv_imgs = F.hflip(pv_imgs)
                art_imgs = F.hflip(art_imgs)

            if pv_mask is not None:
                if isinstance(pv_mask, list):
                    for i in range(len(pv_mask)):
                        pv_mask[i] = F.hflip(pv_mask[i])
                        art_mask[i] = F.hflip(art_mask[i])
                else:
                    pv_mask = F.hflip(pv_mask)
                    art_mask = F.hflip(art_mask)

        return pv_imgs, art_imgs, pv_mask, art_mask


class RandomVerticalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):

        if torch.rand(1) < self.p:
            if isinstance(pv_imgs, list):
                for i in range(len(pv_imgs)):
                    pv_imgs[i] = F.vflip(pv_imgs[i])
                    art_imgs[i] = F.vflip(art_imgs[i])
            else:
                pv_imgs = F.vflip(pv_imgs)
                art_imgs = F.vflip(art_imgs)

            if pv_mask is not None:
                if isinstance(pv_mask, list):
                    for i in range(len(pv_mask)):
                        pv_mask[i] = F.vflip(pv_mask[i])
                        art_mask[i] = F.vflip(art_mask[i])
                else:
                    pv_mask = F.vflip(pv_mask)
                    art_mask = F.vflip(art_mask)

        return pv_imgs, art_imgs, pv_mask, art_mask

class RandomRotation(torch.nn.Module):

    def __init__(self, degrees, p=0.5):
        super().__init__()
        self.p = p
        self.degrees = degrees

    def forward(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):

        if torch.rand(1) < self.p:
            angle = random.randint(self.degrees[0], self.degrees[1])

            if isinstance(pv_imgs, list):
                for i in range(len(pv_imgs)):
                    pv_imgs[i] = pv_imgs[i].rotate(angle, expand=0)
                    art_imgs[i] = art_imgs[i].rotate(angle, expand=0)
            else:
                pv_imgs = pv_imgs
                art_imgs = art_imgs

            if pv_mask is not None:
                if isinstance(pv_mask, list):
                    for i in range(len(pv_mask)):
                        pv_mask[i] = pv_mask[i].rotate(angle, expand=0)
                        art_mask[i] = art_mask[i].rotate(angle, expand=0)
                else:
                    pv_mask = pv_mask.rotate(angle, expand=0)
                    art_mask = art_mask.rotate(angle, expand=0)

        return pv_imgs, art_imgs, pv_mask, art_mask

class CenterCrop(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.crop = T.CenterCrop(size)

    def forward(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):

        if isinstance(pv_imgs, list):
            for i in range(len(pv_imgs)):
                pv_imgs[i] = self.crop(pv_imgs[i])
                art_imgs[i] = self.crop(art_imgs[i])
        else:
            pv_imgs = self.crop(pv_imgs)
            art_imgs = self.crop(art_imgs)

        if pv_mask is not None:
            if isinstance(pv_mask, list):
                for i in range(len(pv_mask)):
                    pv_mask[i] = self.crop(pv_mask[i])
                    art_mask[i] = self.crop(art_mask[i])
            else:
                pv_mask = self.crop(pv_mask)
                art_mask = self.crop(art_mask)

        return pv_imgs, art_imgs, pv_mask, art_mask


class ToTensor_LITS(object):

    def __call__(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):

        flag = False
        if isinstance(pv_imgs, list):
            flag = True
            for i in range(len(pv_imgs)):
                pv_imgs[i] = F.to_tensor(pv_imgs[i])
                art_imgs[i] = F.to_tensor(art_imgs[i])
        else:
            pv_imgs = F.to_tensor(pv_imgs)
            art_imgs = F.to_tensor(art_imgs)

        if pv_mask is not None:
                pv_mask = F.to_tensor(pv_mask)
                art_mask = F.to_tensor(art_mask)
        
        if flag:
            pv_imgs, art_imgs = torch.cat(tuple(pv_imgs)), torch.cat(tuple(art_imgs))

        return pv_imgs, art_imgs, pv_mask, art_mask


class ToTensor(object):

    def __call__(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):

        if isinstance(pv_imgs, list):
            for i in range(len(pv_imgs)):
                pv_imgs[i] = torch.from_numpy(np.array(pv_imgs[i])).unsqueeze(0)
                art_imgs[i] = torch.from_numpy(np.array(art_imgs[i])).unsqueeze(0)
            pv_imgs, art_imgs = torch.cat(tuple(pv_imgs)), torch.cat(tuple(art_imgs))
        else:
            pv_imgs = torch.from_numpy(np.array(pv_imgs)).unsqueeze(0)
            art_imgs = torch.from_numpy(np.array(art_imgs)).unsqueeze(0)

        if pv_mask is not None:
            if isinstance(pv_mask, list):
                for i in range(len(pv_mask)):
                    pv_mask[i] = torch.from_numpy(np.array(pv_mask[i])).unsqueeze(0)
                    art_mask[i] = torch.from_numpy(np.array(art_mask[i])).unsqueeze(0)
                pv_mask, art_mask = torch.cat(tuple(pv_mask)), torch.cat(tuple(art_mask))
            else:
                pv_mask = torch.from_numpy(np.array(pv_mask)).unsqueeze(0)
                art_mask = torch.from_numpy(np.array(art_mask)).unsqueeze(0)

        return pv_imgs, art_imgs, pv_mask, art_mask

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):
        pv_imgs = F.normalize(pv_imgs, self.mean, self.std, inplace=True)
        art_imgs = F.normalize(art_imgs, self.mean, self.std, inplace=True)

        return pv_imgs, art_imgs, pv_mask, art_mask


class Normalize_2d(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):
        for i in range(pv_imgs.size(0)):
            pv_imgs[i] = F.normalize(pv_imgs[i].unsqueeze(0), self.mean, self.std, inplace=True)
            art_imgs[i] = F.normalize(art_imgs[i].unsqueeze(0), self.mean, self.std, inplace=True)

        return pv_imgs, art_imgs, pv_mask, art_mask

class Compose_Test(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):
        for t in self.transforms:
            pv_imgs, art_imgs, pv_mask, art_mask = t(pv_imgs, art_imgs, pv_mask, art_mask)
        return pv_imgs, art_imgs, pv_mask, art_mask


class ToTensor_Test(object):

    def __call__(self, pv_imgs, art_imgs, pv_mask=None, art_mask=None):

        if isinstance(pv_imgs, list):
            for i in range(len(pv_imgs)):
                pv_imgs[i] = F.to_tensor(pv_imgs[i])
                art_imgs[i] = F.to_tensor(art_imgs[i])
                pv_mask[i] = F.to_tensor(pv_mask[i])
                art_mask[i] = F.to_tensor(art_mask[i])


        return pv_imgs, art_imgs, pv_mask, art_mask