import torch.utils.data as data
import torch
import numpy as np
from torchvision import transforms, datasets
import h5py
import random
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        f = h5py.File(file_path,'r')
        self.data_pre = f.get('data_pre')
        self.data_cur = f.get('data_cur')
        self.data_aft = f.get('data_aft')
        self.data_mask = f.get('mask')

        self.label = f.get('label')
        self.data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        return torch.from_numpy(self.data_pre[index, :, :, :].transpose(0,2,1)).float(), \
               torch.from_numpy(self.data_cur[index, :, :, :].transpose(0,2,1)).float(),\
               torch.from_numpy(self.data_aft[index, :, :, :].transpose(0,2,1)).float(),\
               torch.from_numpy(self.data_mask[index, :, :, :].transpose(0,2,1)).float(),\
               torch.from_numpy(self.label[index, :, :, :].transpose(0,2,1)).float()


    def __len__(self):
        assert self.label.shape[0]==self.data_aft.shape[0]

        return self.label.shape[0]

class DatasetFromHdf5_2_data(data.Dataset):
    def __init__(self,data_root_1,data_root_2,transforms=None):
        super(DatasetFromHdf5_2_data, self).__init__()
        f1 = h5py.File(data_root_1,'r')
        f2 = h5py.File(data_root_2,'r')

        self.data_pre = f1.get('data_pre')
        self.data_cur = f1.get('data_cur')
        self.data_aft = f1.get('data_aft')
        self.data_mask = f1.get('mask')
        self.label = f1.get('label')

        self.data_pre_2 = f2.get('data_pre')
        self.data_cur_2 = f2.get('data_cur')
        self.data_aft_2 = f2.get('data_aft')
        self.data_mask_2 = f2.get('mask')
        self.label_2 = f2.get('label')
        self.length = min(self.data_pre.shape[0],self.data_pre_2.shape[0])
    def __getitem__(self, index):
        # print(index)
        # start = random.randint(0,32)
        if index%2==0:
            index = index//2
            return torch.from_numpy(self.data_pre[index, :, ...].transpose(0,2,1)), torch.from_numpy(self.data_cur[index,  :,...].transpose(0,2,1)),\
                   torch.from_numpy(self.data_aft[index, :, ...].transpose(0,2,1)), torch.from_numpy(self.data_mask[index, :, ...].transpose(0,2,1)),torch.from_numpy(self.label[index, :,...].transpose(0,2,1))

        else:
            index = index // 2
            return torch.from_numpy(self.data_pre_2[index, :,...].transpose(0,2,1)), torch.from_numpy(self.data_cur_2[index, :, ...].transpose(0,2,1)), \
                   torch.from_numpy(self.data_aft_2[index, :,...].transpose(0,2,1)), torch.from_numpy(self.data_mask_2[index, :,...].transpose(0,2,1)), torch.from_numpy(self.label_2[index,:,...].transpose(0,2,1))

    def __len__(self):
        return self.length*2


class DatasetFromHdf5_3_data(data.Dataset):
    def __init__(self,data_root_1,data_root_2,data_root_3,transforms=None):
        super(DatasetFromHdf5_3_data, self).__init__()
        f1 = h5py.File(data_root_1,'r')
        f2 = h5py.File(data_root_2,'r')
        f3 = h5py.File(data_root_3,'r')

        self.data_pre = f1.get('data_pre')
        self.data_cur = f1.get('data_cur')
        self.data_aft = f1.get('data_aft')
        self.data_mask = f1.get('mask')
        self.label = f1.get('label')

        self.data_pre_2 = f2.get('data_pre')
        self.data_cur_2 = f2.get('data_cur')
        self.data_aft_2 = f2.get('data_aft')
        self.data_mask_2 = f2.get('mask')
        self.label_2 = f2.get('label')

        self.data_pre_3 = f3.get('data_pre')
        self.data_cur_3 = f3.get('data_cur')
        self.data_aft_3 = f3.get('data_aft')
        self.data_mask_3 = f3.get('mask')
        self.label_3 = f3.get('label')

        self.length = min(self.data_pre.shape[0],self.data_pre_2.shape[0],self.data_pre_3.shape[0])
    def __getitem__(self, index):
        # print(index)
        
        if index%3==0:
            index = index//3
            return torch.from_numpy(self.data_pre[index, :, ...].transpose(0,2,1)), torch.from_numpy(self.data_cur[index,  :, ...].transpose(0,2,1)),\
                   torch.from_numpy(self.data_aft[index, :, ...].transpose(0,2,1)), torch.from_numpy(self.data_mask[index, :, ...].transpose(0,2,1)),torch.from_numpy(self.label[index, :,...].transpose(0,2,1))

        elif index%3==1:
            index = index //3
            return torch.from_numpy(self.data_pre_2[index, :,...].transpose(0,2,1)), torch.from_numpy(self.data_cur_2[index, :, ...].transpose(0,2,1)), \
                   torch.from_numpy(self.data_aft_2[index, :,...].transpose(0,2,1)), torch.from_numpy(self.data_mask_2[index, :, ...].transpose(0,2,1)), torch.from_numpy(self.label_2[index,:,...].transpose(0,2,1))

        else:
            index = index //3
            return torch.from_numpy(self.data_pre_3[index, :,...].transpose(0,2,1)), torch.from_numpy(self.data_cur_3[index, :, ...].transpose(0,2,1)), \
                   torch.from_numpy(self.data_aft_3[index, :,...].transpose(0,2,1)), torch.from_numpy(self.data_mask_3[index, :, ...].transpose(0,2,1)), torch.from_numpy(self.label_3[index,:,...].transpose(0,2,1))


    def __len__(self):
        return self.length*3
