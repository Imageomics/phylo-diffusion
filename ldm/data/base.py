from abc import abstractmethod
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset



class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, horizontalflip=False, random_contrast=False, shiftrotate=False, labels=None, unique_skipped_labels=[]):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        
        self.labels_without_skipped = None
        if len(unique_skipped_labels)!=0:
            self.labels_without_skipped = dict()
            for i in self.labels.keys():
                self.labels_without_skipped[i] = [a for indx, a in enumerate(labels[i]) if labels['class'][indx] not in unique_skipped_labels]
            self._length = len(self.labels_without_skipped['class'])

        
        

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            l = [self.rescaler ]
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            l.append(self.cropper)
            if horizontalflip==True:
                l.append(albumentations.HorizontalFlip(p=0.2))
            if shiftrotate==True:
                l.append(albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=0, value=( int(0.485*255), int(0.456*255), int(0.406*255 )), p=0.3))
            if random_contrast==True:
                l.append(albumentations.RandomBrightnessContrast(p=0.3))
            self.preprocessor = albumentations.Compose(l)
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def __getitem__(self, i):
        labels = self.labels if self.labels_without_skipped is None else self.labels_without_skipped
        example = dict()
        example["image"] = self.preprocess_image(labels["file_path_"][i])
        for k in labels:
            example[k] = labels[k][i]
        return example