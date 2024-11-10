import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt

from torchvision.models import resnet50,ResNet50_Weights
from torchvision.models import swin_b, Swin_B_Weights

from io import open
from submodules import *
import os

from tqdm import tqdm

# GPU 설정
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

captions_list

#create Vocab objects for each language
vocab=Vocab()

#build the vocab from caption_list
for caption in captions_list:
    vocab.buildVocab(caption)

#print vocab size
# print("Vocab Length:",vocab.nwords)
# print()
# print(vocab.word2index)
# print(vocab.index2word)
# print()
# print(vocab.word2count)
    
max_cap_length=73   # 데이터셋의 Caption의 길이가 다르기 때문에 최대 길이를 설정함. max_cap_length보다 짧은 caption은 0으로 padding을 수행해야함.

dataset=CustomDataset(images_dir_path, img_filenames_list, captions_list, vocab, max_cap_length)
train_dataset,test_dataset=random_split(dataset,[0.999,0.001])

batch_size=64
train_dataloader=DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset,batch_size=1, shuffle=False)

# 데이터 수 확인
# print('Total dataset :',len(dataset))
# print('Train dataset :',len(train_dataset))
# print('Test dataset : ',len(test_dataset))

# shape 확인
# for idx, i in enumerate(train_dataloader):
#     print('Index',idx)
#     print(i[0].shape) # torch.Size([64, 3, 224, 224])
#     print(i[1].shape) # torch.Size([64, 75]) 
#     break
    
pretrained_feature_extractor=swin_b(weights = Swin_B_Weights)
# pretrained_feature_extractor=resnet50(pretrained=True)
# pretrained_feature_extractor.head=nn.Linear(1000,1024)

encoder=Encoder(pretrained_feature_extractor).to(device)
    
embed_size=300
hidden_size=1000

# vocab.nwords : (8446) 임베딩을 할 단어들의 개수, 즉, 단어 집합의 크기
# embed_size : 300
decoder=Decoder(vocab.nwords,embed_size,hidden_size).to(device)

def train_one_epoch(epoch):
    encoder.train()
    decoder.train()
    track_loss=0

    # 배치 사이즈만큼 데이터를 가져옴.
    for idx, (imgs,caption) in tqdm(enumerate(train_dataloader)):
        # imgs size : torch.Size([64, 3, 224, 224])
        # caption size : torch.Size([64, 75])
        imgs=imgs.to(device)
        caption=caption.to(device)
        # extracted_features : [64, 1024]
        extracted_features=encoder(imgs)
        # decoder_hidden : 이미지 feature.
        # extracted_features : [64, 1024] ->  decoder_hidden: [1, 64, 1024]
        decoder_hidden=torch.reshape(extracted_features,(1,extracted_features.shape[0],-1))

        # caption이랑, 이미지 feature가 들어감.
        yhats, decoder_hidden = decoder(caption[:,0:-1],decoder_hidden)

        # gt는 caption에서 첫번째 단어를 제외한 것. [64, 74]
        gt=caption[:,1:]

        # ([64, 74, 8446])-> [4736, 8446] 
        yhats_reshaped=yhats.view(-1,yhats.shape[-1])

        # [64, 74] -> [64*74] = [4736]
        gt=gt.reshape(-1)

        # NLLLoss는 log_softmax를 적용한 후, negative log likelihood loss를 계산함.
        # yhats_reshaped = [4736, 8446] , gt = [4736]
        loss=loss_fn(yhats_reshaped,gt)
        track_loss+=loss.item()

        opte.zero_grad()
        optd.zero_grad()
        
        loss.backward()
        
        opte.step()
        optd.step()

        if not os.path.exists('state_dict'):
            os.makedirs('state_dict')        

        encoder_path = f"Python/ImageCaptioning/state_dict/encoder_epoch_{epoch}.pt"
        decoder_path = f"Python/ImageCaptioning/state_dict/decoder_epoch_{epoch}.pt"

        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)        

        if idx%50==0:
            print("Mini Batch=", idx+1," Running Loss=",track_loss/(idx+1), sep="")
        
    return track_loss/len(train_dataloader)

#eval loop (written assuming batch_size=1)
def eval_one_epoch():
    encoder.eval()
    decoder.eval()
    track_loss=0

    with torch.no_grad():
        for i, (imgs,t_ids) in enumerate(test_dataloader):

            imgs=imgs.to(device)
            t_ids=t_ids.to(device)
            # t_ids size : torch.Size([1, 75])
            # imgs size : torch.Size([1, 3, 224, 224])
            print(t_ids.size())
            # extracted_features = [1, 1024]
            extracted_features=encoder(imgs)
            # extracted_features = [1, 1024] -> decoder_hidden = [1, 1, 1024]
            decoder_hidden=torch.reshape(extracted_features,(1,extracted_features.shape[0],-1)) 
            # print(decoder_hidden.size())
            # 배치 사이즈의 모든 행들 중에 첫번째 열만 가져옴.
            input_ids=t_ids[:,0] # SOS 추출. # input_ids.size() : torch.Size([1])
            # print('1. ',input_ids.size())
            yhats=[]
            pred_sentence=""

            for j in range(1, max_cap_length+1): # j start from 1 because 0th token is SOS token
                probs, decoder_hidden = decoder(input_ids.unsqueeze(0),decoder_hidden)
                yhats.append(probs)
                _,input_ids=torch.topk(probs,1, dim=-1)
                # print('2',input_ids.size()) # [1, 1, 1]
                input_ids=input_ids.squeeze(1)
                input_ids=input_ids.squeeze(1)
                word=vocab.index2word[input_ids.item()]
                pred_sentence+=word+" "
                if input_ids.item()==1: # batch_size=1 , 1=EOS (End of Sentence token)
                    break
    
            # t_ids = 길이가 75인 index값으로 구성된 리스트(리스트 길이 75).
            gt_sentence=ids2Sentence(t_ids,vocab)
            
            print("Input Image:")
            img=imgs[0]
            img[0]=(img[0]*0.229)+0.485
            img[1]=(img[1]*0.224)+0.456
            img[2]=(img[2]*0.225)+0.406
            plt.imshow(torch.permute(imgs[0],(1,2,0)).detach().cpu())
            plt.show()
            
            print("GT Sentence:",gt_sentence)
            
            print("Predicted Sentence:",pred_sentence)
            
            yhats_cat=torch.cat(yhats,dim=1)
            yhats_reshaped=yhats_cat.view(-1,yhats_cat.shape[-1])
            gt=t_ids[:,1:j+1]
            gt=gt.view(-1)
            

            loss=loss_fn(yhats_reshaped,gt)
            track_loss+=loss.item()
            
            
        print("-----------------------------------")
        return track_loss/len(test_dataloader)
    
loss_fn=nn.NLLLoss(ignore_index=0).to(device)
lr=0.001

optd=optim.Adam(params=decoder.parameters(), lr=lr)
opte=optim.Adam(params=encoder.parameters(), lr=lr)

n_epochs=1

for e in range(n_epochs):
    print("Epoch=",e+1, " Loss=", round(train_one_epoch(e),4), sep="")

for e in range(1):
    print("Epoch=",e+1, " Loss=", round(eval_one_epoch(),4), sep="")
