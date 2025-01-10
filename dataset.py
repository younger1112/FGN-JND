from os import listdir
from os.path import join

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.images = [join(self.a_path, x) for x in self.image_filenames]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        # 不调整图像大小
        # a = a.resize((256, 256), Image.BICUBIC)  
        # b = b.resize((256, 256), Image.BICUBIC)  
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)


""" class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        #self.c_path = join(image_dir, "c")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.images = [join(self.a_path, x) for x in self.image_filenames]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = Image.open(self.images[index]).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        #c = Image.open(join(self.c_path, self.image_filenames[index])).convert('RGB')

        a = self.transform(a)
        b = self.transform(b)
        #c = self.transform(c)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a
    def __len__(self):
        return len(self.image_filenames)
 """