import unicodedata
import re
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
import torch
import torch.nn as nn
from PIL import Image

# unicode 2 ascii, remove non-letter characters, trim
# 단어토큰화를 수행하기 때문에 단어가 아닌것들은 제거 하는 코드.
def normalizeString(s): 
    sres=""
    for ch in unicodedata.normalize('NFD', s): 
        #Return the normal form form ('NFD') for the Unicode string s.
        if unicodedata.category(ch) != 'Mn':
            # The function in the first part returns the general 
            # category assigned to the character ch as string. 
            # "Mn' refers to Mark, Nonspacing
            sres+=ch
    #sres = re.sub(r"([.!?])", r" \1", sres) 
    # inserts a space before any occurrence of ".", "!", or "?" in the string sres. 
    sres = re.sub(r"[^a-zA-Z!?,]+", r" ", sres) 
    # this line of code replaces any sequence of characters in sres 
    # that are not letters (a-z or A-Z) or the punctuation marks 
    # "!", "," or "?" with a single space character.
    return sres.strip()

# 주어진 문장을 기반으로 어휘 사전을 구축하는 class 로. / SOS, EOS , 중복되지 않도록 단어들을 추가함.
class Vocab:
    def __init__(self):
        self.word2index={'SOS':0, 'EOS':1}
        self.index2word={0:'SOS', 1:'EOS'}
        self.word2count={}
        self.nwords=2
    
    def buildVocab(self,s):
        for word in s.split(" "):
            # 단어(Word)가 없을 때만 if문을 돔.
            if word not in self.word2index:
                self.word2index[word]=self.nwords # word2index에 word를 key로, nwords를 value로 추가
                self.index2word[self.nwords]=word # index2word에 nwords를 key로, word를 value로 추가
                self.word2count[word]=1 # word2count는 단어가 나온 횟수를 저장함.
                self.nwords+=1  # nwords = number of words
            else:
                self.word2count[word]+=1

class CustomDataset(Dataset):
    def __init__(self,images_dir_path, img_filenames_list, captions_list, vocab, max_cap_length):
        super().__init__()
        self.images_dir_path=images_dir_path
        self.img_filenames_list=img_filenames_list
        self.captions_list=captions_list
        self.length=len(self.captions_list)
        self.transform=Compose([Resize((224,224), antialias=True),
                                ToTensor(), 
                                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.vocab=vocab
        self.max_cap_length=max_cap_length
    
    def __len__(self):
        return self.length
    
    # 각 토큰을 고유한 숫자 식별자로 매핑하는 함수.
    def get_input_ids(self, sentence,vocab):
        input_ids=[0]*(self.max_cap_length+1)
        i=0
        # 단어 토큰화라서 Split 함수(공백을 기준으로 토큰화 진행) 적용.
        for word in sentence.split(" "):
            input_ids[i]=vocab.word2index[word]
            i=i+1

        input_ids.insert(0,vocab.word2index['SOS'])
        i=i+1
        input_ids[i]=vocab.word2index['EOS']

        # torch 형태로 변환함.
        return torch.tensor(input_ids)
    
    def __getitem__(self,idx):
        imgfname,caption=self.img_filenames_list[idx],self.captions_list[idx]
        
        imgfname=self.images_dir_path+imgfname
        img=Image.open(imgfname)
        img=self.transform(img)
        
        caption=self.get_input_ids(caption,self.vocab)       
        
        return img,caption
    
## Encoder 구현
class Encoder(nn.Module):
    def __init__(self, pretrained_feature_extractor):
        super().__init__()
        self.pretrained_feature_extractor=pretrained_feature_extractor
        
    def forward(self,x):
        features=self.pretrained_feature_extractor(x)

        for name, param in self.pretrained_feature_extractor.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                    
        return features
    
class Decoder(nn.Module):
    def __init__(self,output_size,embed_size,hidden_size):
        super().__init__()
        # num_embeddings :임베딩을 할 단어들의 개수, 즉, 단어 집합의 크기
        # embeding_dim : 임베딩 할 벡터의 차원, 사용자가 정의 하는 하이퍼 파라미터.
        # padding_idx : 패딩을 위한 토큰의 인덱스 알려줌. (선택적으로 사용함)
        self.e=nn.Embedding(num_embeddings=output_size,
                            embedding_dim=embed_size)
        self.relu=nn.ReLU()
        self.gru=nn.GRU(embed_size, hidden_size, batch_first=True)
        self.lin=nn.Linear(hidden_size,output_size)
        self.lsoftmax=nn.LogSoftmax(dim=-1)
    
    def forward(self,x,prev_hidden):
        x=self.e(x)
        x=self.relu(x)
        output,hidden=self.gru(x,prev_hidden)
        y=self.lin(output)
        y=self.lsoftmax(y)
        return y, hidden
    
# 단어의 id를 문장으로 변환하는 함수.
def ids2Sentence(ids,vocab):
    sentence=""
    for id in ids.squeeze():
        if id==0:  # id=0은 SOS이므로, continue
            continue
        word=vocab.index2word[id.item()]
        sentence+=word + " "
        if id==1:  # id=1은 EOS이므로, break
            break
    return sentence
