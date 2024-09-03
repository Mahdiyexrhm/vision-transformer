from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from base import BaseDataLoader
from shutil import copyfile
import os


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class FER(BaseDataLoader):
    """
    FER-2013 data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5077, 0.5077, 0.5077), (0.2120, 0.2120, 0.2120))
        ])
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, "fer2013", 'train')
        self.train_set = datasets.ImageFolder(root=self.train_path, transform=trsfm)
        super().__init__(self.train_set, batch_size, shuffle, validation_split, num_workers)


class Stanford40(BaseDataLoader):
    """
    Stanford40 data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        creating_dataset()
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=3, magnitude=15),  # RandAugment با تعداد عملیات و شدت بیشتر
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_set = datasets.ImageFolder(root='StanfordActionDataset/train/',
                                              transform=transform_train)
        super().__init__(self.train_set, batch_size, shuffle, validation_split, num_workers)


def creating_dataset():
    images_path = "./JPEGImages"
    labels_path = "./ImageSplits"
    new_dataset_path = "./StanfordActionDataset"

    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)
        os.makedirs(os.path.join(new_dataset_path, 'train'))
        os.makedirs(os.path.join(new_dataset_path, 'test'))

    txts = os.listdir(labels_path)
    for txt in txts:
        idx = txt.rfind('_')
        class_name = txt[:idx]
        if class_name in ['actions.tx', 'test.tx', 'train.tx']:
            continue
        train_or_test = txt[idx + 1:-4]
        txt_contents = open(os.path.join(labels_path, txt)).read()
        image_names = txt_contents.split('\n')
        for image_name in image_names[:-1]:
            class_path = os.path.join(new_dataset_path, train_or_test, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            copyfile(os.path.join(images_path, image_name),
                     os.path.join(class_path, image_name))

 
    
                    
