from submodules import *
from torchvision.models import resnet50,ResNet50_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import swin_b, Swin_B_Weights
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
import json

# GPU 설정
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

capt_file_path=r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/flickr8k/captions.txt'
images_dir_path=r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/flickr8k/Images/'
epoch = 9

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

batch_size=64
train_dataloader=DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset,batch_size=1, shuffle=False)

embed_size=300
hidden_size=1000

pretrained_feature_extractor=resnet50(weights=ResNet50_Weights.DEFAULT)
pretrained_feature_extractor.fc=nn.Linear(2048,1000)
pretrained_feature_extractor1=vgg16(weights=VGG16_Weights.DEFAULT)
pretrained_feature_extractor1.fc=nn.Linear(2048,1000)
pretrained_feature_extractor2=vit_b_16(weights = ViT_B_16_Weights.DEFAULT)
pretrained_feature_extractor3=swin_b(weights = Swin_B_Weights)

encoder=Encoder(pretrained_feature_extractor).to(device)
decoder=Decoder(vocab.nwords,embed_size,hidden_size).to(device)
encoder1=Encoder(pretrained_feature_extractor1).to(device)
decoder1=Decoder(vocab.nwords,embed_size,hidden_size).to(device)
encoder2=Encoder(pretrained_feature_extractor2).to(device)
decoder2=Decoder(vocab.nwords,embed_size,hidden_size).to(device)
encoder3=Encoder(pretrained_feature_extractor3).to(device)
decoder3=Decoder(vocab.nwords,embed_size,hidden_size).to(device)

encoder.load_state_dict(torch.load(f'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/state_dict/res/encoder_epoch_{epoch}.pt'))
decoder.load_state_dict(torch.load(f'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/state_dict/res/decoder_epoch_{epoch}.pt'))
encoder1.load_state_dict(torch.load(f'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/state_dict/vgg/encoder_epoch_{epoch}.pt'))
decoder1.load_state_dict(torch.load(f'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/state_dict/vgg/decoder_epoch_{epoch}.pt'))
encoder2.load_state_dict(torch.load(f'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/state_dict/vit/encoder_epoch_{epoch}.pt'))
decoder2.load_state_dict(torch.load(f'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/state_dict/vit/decoder_epoch_{epoch}.pt'))
encoder3.load_state_dict(torch.load(f'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/state_dict/swin/encoder_epoch_{epoch}.pt'))
decoder3.load_state_dict(torch.load(f'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/state_dict/swin/decoder_epoch_{epoch}.pt'))

#eval loop (written assuming batch_size=1)
def eval_one_epoch():
    encoder.eval()
    decoder.eval()
    track_loss=0

    with torch.no_grad():
        ref = {'images': [], 'annotations': []}
        cap = []

        for i, (imgs,t_ids) in enumerate(test_dataloader):

            imgs=imgs.to(device)
            t_ids=t_ids.to(device)
            # t_ids size : torch.Size([1, 75])
            # imgs size : torch.Size([1, 3, 224, 224])
            # print(t_ids.size())
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
            
            # print("Input Image:")
            # img=imgs[0]
            # img[0]=(img[0]*0.229)+0.485
            # img[1]=(img[1]*0.224)+0.456
            # img[2]=(img[2]*0.225)+0.406
            # plt.imshow(torch.permute(imgs[0],(1,2,0)).detach().cpu())
            # plt.show()
            
            # print("GT Sentence:",gt_sentence)
            ref['images'].append({'id': i})
            ref['annotations'].append({'image_id': i, 'id': i, 'caption': gt_sentence})
            
            # print("Predicted Sentence:",pred_sentence)
            cap.append({'image_id': i, 'caption': pred_sentence})
            
            yhats_cat=torch.cat(yhats,dim=1)
            yhats_reshaped=yhats_cat.view(-1,yhats_cat.shape[-1])
            gt=t_ids[:,1:j+1]
            gt=gt.view(-1)
            
            loss=loss_fn(yhats_reshaped,gt)
            track_loss+=loss.item()
            
        with open(r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/res/' + f'references_res{epoch}.json', 'w') as fgts:
            json.dump(ref, fgts)
        with open(r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/res/' + f'captions_res{epoch}.json', 'w') as fres:
            json.dump(cap, fres)
        print("-----------------------------------")
        return track_loss/len(test_dataloader)
    
def eval_one_epoch1():
    encoder1.eval()
    decoder1.eval()
    track_loss=0

    with torch.no_grad():
        ref = {'images': [], 'annotations': []}
        cap = []

        for i, (imgs,t_ids) in enumerate(test_dataloader):

            imgs=imgs.to(device)
            t_ids=t_ids.to(device)
            # t_ids size : torch.Size([1, 75])
            # imgs size : torch.Size([1, 3, 224, 224])
            # print(t_ids.size())
            # extracted_features = [1, 1024]
            extracted_features=encoder1(imgs)
            # extracted_features = [1, 1024] -> decoder_hidden = [1, 1, 1024]
            decoder_hidden=torch.reshape(extracted_features,(1,extracted_features.shape[0],-1)) 
            # print(decoder_hidden.size())
            # 배치 사이즈의 모든 행들 중에 첫번째 열만 가져옴.
            input_ids=t_ids[:,0] # SOS 추출. # input_ids.size() : torch.Size([1])
            # print('1. ',input_ids.size())
            yhats=[]
            pred_sentence=""

            for j in range(1, max_cap_length+1): # j start from 1 because 0th token is SOS token
                probs, decoder_hidden = decoder1(input_ids.unsqueeze(0),decoder_hidden)
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
            
            # print("Input Image:")
            # img=imgs[0]
            # img[0]=(img[0]*0.229)+0.485
            # img[1]=(img[1]*0.224)+0.456
            # img[2]=(img[2]*0.225)+0.406
            # plt.imshow(torch.permute(imgs[0],(1,2,0)).detach().cpu())
            # plt.show()
            
            # print("GT Sentence:",gt_sentence)
            ref['images'].append({'id': i})
            ref['annotations'].append({'image_id': i, 'id': i, 'caption': gt_sentence})
            
            # print("Predicted Sentence:",pred_sentence)
            cap.append({'image_id': i, 'caption': pred_sentence})
            
            yhats_cat=torch.cat(yhats,dim=1)
            yhats_reshaped=yhats_cat.view(-1,yhats_cat.shape[-1])
            gt=t_ids[:,1:j+1]
            gt=gt.view(-1)
            
            loss=loss_fn(yhats_reshaped,gt)
            track_loss+=loss.item()
            
        with open(r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/vgg/' + f'references_vgg{epoch}.json', 'w') as fgts:
            json.dump(ref, fgts)
        with open(r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/vgg/' + f'captions_vgg{epoch}.json', 'w') as fres:
            json.dump(cap, fres)
        print("-----------------------------------")
        return track_loss/len(test_dataloader)
    
def eval_one_epoch2():
    encoder2.eval()
    decoder2.eval()
    track_loss=0

    with torch.no_grad():
        ref = {'images': [], 'annotations': []}
        cap = []

        for i, (imgs,t_ids) in enumerate(test_dataloader):

            imgs=imgs.to(device)
            t_ids=t_ids.to(device)
            # t_ids size : torch.Size([1, 75])
            # imgs size : torch.Size([1, 3, 224, 224])
            # print(t_ids.size())
            # extracted_features = [1, 1024]
            extracted_features=encoder2(imgs)
            # extracted_features = [1, 1024] -> decoder_hidden = [1, 1, 1024]
            decoder_hidden=torch.reshape(extracted_features,(1,extracted_features.shape[0],-1)) 
            # print(decoder_hidden.size())
            # 배치 사이즈의 모든 행들 중에 첫번째 열만 가져옴.
            input_ids=t_ids[:,0] # SOS 추출. # input_ids.size() : torch.Size([1])
            # print('1. ',input_ids.size())
            yhats=[]
            pred_sentence=""

            for j in range(1, max_cap_length+1): # j start from 1 because 0th token is SOS token
                probs, decoder_hidden = decoder2(input_ids.unsqueeze(0),decoder_hidden)
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
            
            # print("Input Image:")
            # img=imgs[0]
            # img[0]=(img[0]*0.229)+0.485
            # img[1]=(img[1]*0.224)+0.456
            # img[2]=(img[2]*0.225)+0.406
            # plt.imshow(torch.permute(imgs[0],(1,2,0)).detach().cpu())
            # plt.show()
            
            # print("GT Sentence:",gt_sentence)
            ref['images'].append({'id': i})
            ref['annotations'].append({'image_id': i, 'id': i, 'caption': gt_sentence})
            
            # print("Predicted Sentence:",pred_sentence)
            cap.append({'image_id': i, 'caption': pred_sentence})
            
            yhats_cat=torch.cat(yhats,dim=1)
            yhats_reshaped=yhats_cat.view(-1,yhats_cat.shape[-1])
            gt=t_ids[:,1:j+1]
            gt=gt.view(-1)
            
            loss=loss_fn(yhats_reshaped,gt)
            track_loss+=loss.item()
            
        with open(r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/vit/' + f'references_vit{epoch}.json', 'w') as fgts:
            json.dump(ref, fgts)
        with open(r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/vit/' + f'captions_vit{epoch}.json', 'w') as fres:
            json.dump(cap, fres)
        print("-----------------------------------")
        return track_loss/len(test_dataloader)
    
def eval_one_epoch3():
    encoder3.eval()
    decoder3.eval()
    track_loss=0

    with torch.no_grad():
        ref = {'images': [], 'annotations': []}
        cap = []

        for i, (imgs,t_ids) in enumerate(test_dataloader):

            imgs=imgs.to(device)
            t_ids=t_ids.to(device)
            # t_ids size : torch.Size([1, 75])
            # imgs size : torch.Size([1, 3, 224, 224])
            # print(t_ids.size())
            # extracted_features = [1, 1024]
            extracted_features=encoder3(imgs)
            # extracted_features = [1, 1024] -> decoder_hidden = [1, 1, 1024]
            decoder_hidden=torch.reshape(extracted_features,(1,extracted_features.shape[0],-1)) 
            # print(decoder_hidden.size())
            # 배치 사이즈의 모든 행들 중에 첫번째 열만 가져옴.
            input_ids=t_ids[:,0] # SOS 추출. # input_ids.size() : torch.Size([1])
            # print('1. ',input_ids.size())
            yhats=[]
            pred_sentence=""

            for j in range(1, max_cap_length+1): # j start from 1 because 0th token is SOS token
                probs, decoder_hidden = decoder3(input_ids.unsqueeze(0),decoder_hidden)
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
            
            # print("Input Image:")
            # img=imgs[0]
            # img[0]=(img[0]*0.229)+0.485
            # img[1]=(img[1]*0.224)+0.456
            # img[2]=(img[2]*0.225)+0.406
            # plt.imshow(torch.permute(imgs[0],(1,2,0)).detach().cpu())
            # plt.show()
            
            # print("GT Sentence:",gt_sentence)
            ref['images'].append({'id': i})
            ref['annotations'].append({'image_id': i, 'id': i, 'caption': gt_sentence})
            
            # print("Predicted Sentence:",pred_sentence)
            cap.append({'image_id': i, 'caption': pred_sentence})
            
            yhats_cat=torch.cat(yhats,dim=1)
            yhats_reshaped=yhats_cat.view(-1,yhats_cat.shape[-1])
            gt=t_ids[:,1:j+1]
            gt=gt.view(-1)
            
            loss=loss_fn(yhats_reshaped,gt)
            track_loss+=loss.item()
            
        with open(r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/swin/' + f'references_swin{epoch}.json', 'w') as fgts:
            json.dump(ref, fgts)
        with open(r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/swin/' + f'captions_swin{epoch}.json', 'w') as fres:
            json.dump(cap, fres)
        print("-----------------------------------")
        return track_loss/len(test_dataloader)

loss_fn=nn.NLLLoss(ignore_index=0).to(device)

for e in range(1):
    loss = eval_one_epoch()
    print("Epoch=",e+1, " Loss=", round(loss,4), sep="")

for e in range(1):
    loss = eval_one_epoch1()
    print("Epoch=",e+1, " Loss=", round(loss,4), sep="")

for e in range(1):
    loss = eval_one_epoch2()
    print("Epoch=",e+1, " Loss=", round(loss,4), sep="")

for e in range(1):
    loss = eval_one_epoch3()
    print("Epoch=",e+1, " Loss=", round(loss,4), sep="")
