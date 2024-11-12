import matplotlib.pyplot as plt
import numpy as np

savepath = r'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/Image-Captioning/results/'

x = np.array([1, 2, 3, 4, 5, 6])

y_resnet_b1 = np.array([0.378, 0.404, 0.401, 0.431, 0.465, 0.455])
y_resnet_b4 = np.array([0.053, 0.115, 0.107, 0.136, 0.115, 0.157])
y_resnet_meteor = np.array([0.177, 0.182, 0.196, 0.217, 0.218, 0.22])

y_swin_b1 = np.array([0.368, 0.406, 0.429, 0.392, 0.455, 0.468])
y_swin_b4 = np.array([0.056, 0.096, 0.138, 0.123, 0.122, 0.184])
y_swin_meteor = np.array([0.158, 0.201, 0.209, 0.191, 0.216, 0.238])

plt.plot(x, y_swin_b1, label = 'swin_BLEU1')
plt.plot(x, y_resnet_b1, label = 'resnet50_BLEU1')

plt.plot(x, y_swin_meteor, label = 'swin_METEOR')
plt.plot(x, y_resnet_meteor, label = 'resnet50_METEOR')

plt.plot(x, y_swin_b4, label = 'swin_BLEU4')
plt.plot(x, y_resnet_b4, label = 'resnet50_BLEU4')

plt.legend()
plt.xticks(x)
plt.xlabel('epoch')
plt.ylabel('score')
plt.title('Evaluation Scores')
plt.savefig(savepath + 'Evaluation Scores_6epoch' + '.png')
