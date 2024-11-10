from sklearn.manifold import TSNE # sklearn 사용하면 easy !! 
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from submodules import *
from torchvision.models import resnet50,ResNet50_Weights
from torchvision.models import swin_b, Swin_B_Weights
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

capt_file_path='Python/flickr8k/captions.txt'
images_dir_path='Python/flickr8k/Images/'

#read data
data=open(capt_file_path).read().strip().split('\n')
data=data[1:]

img_filenames_list=[]
captions_list=[]

# caption 만 추출함.
for s in data:
    templist=s.lower().split(",")
    img_path=templist[0]
    caption=",".join(s for s in templist[1:])
    caption=normalizeString(caption)
    img_filenames_list.append(img_path)
    captions_list.append(caption)

#create Vocab objects for each language
vocab=Vocab()
#build the vocab from caption_list
for caption in captions_list:
    vocab.buildVocab(caption)

max_cap_length=73   # 데이터셋의 Caption의 길이가 다르기 때문에 최대 길이를 설정함. max_cap_length보다 짧은 caption은 0으로 padding을 수행해야함.

dataset=CustomDataset(images_dir_path, img_filenames_list, captions_list, vocab, max_cap_length)
train_dataset,test_dataset=random_split(dataset,[0.999,0.001])

test_dataloader=DataLoader(dataset=test_dataset,batch_size=1, shuffle=False)

embed_size=300
hidden_size=1000

pretrained_feature_extractor=swin_b(weights = Swin_B_Weights)
encoder=Encoder(pretrained_feature_extractor).to(device)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

encoder.fc = Identity()

actual = []
deep_features = []

encoder.eval()
with torch.no_grad():
    for i, (imgs,t_ids) in enumerate(test_dataloader):
        imgs=imgs.to(device)
        t_ids=t_ids.to(device)

        extracted_features=encoder(imgs)

        images, labels = data[0].to(device), data[1].to(device)
        features = encoder(images) # 512 차원

        deep_features += features.cpu().numpy().tolist()
        actual += labels.cpu().numpy().tolist()

tsne = TSNE(n_components=2, random_state=0) # 사실 easy 함 sklearn 사용하니..
cluster = np.array(tsne.fit_transform(np.array(deep_features)))
actual = np.array(actual)

plt.figure(figsize=(10, 10))
cifar = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i, label in zip(range(10), cifar):
    idx = np.where(actual == i)
    plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)

plt.legend()
plt.show()
