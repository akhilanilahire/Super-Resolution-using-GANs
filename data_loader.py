from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, ToPILImage, CenterCrop, Resize

def highres_transform_img(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
        Normalize([0, 0, 0], [1, 1, 1])
    ])


def lowres_transform_img(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor(),
        Normalize([0, 0, 0], [1, 1, 1])
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

class TrainDataset_loader(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDataset_loader, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir)]
        self.highres_transform = highres_transform_img(crop_size)
        self.lowres_transform = lowres_transform_img(crop_size, upscale_factor)

    def __getitem__(self, index):
        highres_image = self.highres_transform(Image.open(self.image_filenames[index]))
        lowres_image = self.lowres_transform(highres_image)
        return lowres_image, highres_image

    def __len__(self):
        return len(self.image_filenames)


class ValDataset_loader(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDataset_loader, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir)[:5]] #if check_image_file(x)]


    def __getitem__(self, index):

        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = 256
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = hr_scale(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = CenterCrop(crop_size//4)(lr_image)
        hr_restore_img = CenterCrop(crop_size)(hr_restore_img)
        # print(f"crop_size: {crop_size}, w: {w}, h: {h}, low: ({lr_image.width},{lr_image.height}), Bicubic: ({hr_restore_img.width},{hr_restore_img.height}), high: ({hr_image.width},{hr_image.height})")

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
    def __len__(self):
        return len(self.image_filenames)

class TestDataset_loader(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDataset_loader, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)