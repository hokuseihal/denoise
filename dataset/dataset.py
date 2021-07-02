import os
from PIL import Image

class ImageFolderDataset():
    def __init__(self, folderpath, transform, target_transform):
        super(ImageFolderDataset, self).__init__()
        self.folderpath=folderpath
        self.transform = transform
        self.target_transform = target_transform
        self.imglist = os.listdir(folderpath)
        assert len(self.imglist) != 0, f"No images in {folderpath}"

    def __getitem__(self, idx):
        im=Image.open(os.path.join(self.folderpath,self.imglist[idx]))
        return self.transform(im),self.target_transform(im)
    def __len__(self):
        return len(self.imglist)
