import matplotlib.pyplot as plt
import numpy as np

savepath = r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/results/'

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

y_resnet_b1 = np.array([0.378, 0.404, 0.401, 0.431, 0.465, 0.455, 0.445, 0.532, 0.53, 0.546])
y_resnet_b4 = np.array([0.053, 0.115, 0.107, 0.136, 0.115, 0.157, 0.144, 0.24, 0.219, 0.236])
y_resnet_meteor = np.array([0.177, 0.182, 0.196, 0.217, 0.218, 0.22, 0.21, 0.265, 0.271, 0.274])

y_swin_b1 = np.array([0.368, 0.406, 0.429, 0.392, 0.455, 0.468, 0.434, 0.501, 0.53, 0.45])
y_swin_b4 = np.array([0.056, 0.096, 0.138, 0.123, 0.122, 0.184, 0.148, 0.2, 0.198, 0.15])
y_swin_meteor = np.array([0.158, 0.201, 0.209, 0.191, 0.216, 0.238, 0.214, 0.244, 0.263, 0.22])

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
plt.savefig(savepath + 'Evaluation Scores' + '.png')
