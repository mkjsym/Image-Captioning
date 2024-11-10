import matplotlib.pyplot as plt
import numpy as np

savepath = 'GitHub/coding-test/YM/temp/'

x = np.array([1, 2, 3])

y_resnet_b1 = np.array([0.378, 0.404, 0.401])
y_resnet_b4 = np.array([0.053, 0.115, 0.107])
y_resnet_meteor = np.array([0.177, 0.182, 0.196])

y_swin_b1 = np.array([0.368, 0.406, 0.429])
y_swin_b4 = np.array([0.056, 0.096, 0.138])
y_swin_meteor = np.array([0.158, 0.201, 0.209])

plt.plot(x, y_swin_b1, label = 'swin_b1')
plt.plot(x, y_resnet_b1, label = 'resnet50_b1')

plt.plot(x, y_swin_meteor, label = 'swin_meteor')
plt.plot(x, y_resnet_meteor, label = 'resnet50_meteor')

plt.plot(x, y_swin_b4, label = 'swin_b4')
plt.plot(x, y_resnet_b4, label = 'resnet50_b4')

plt.legend()
plt.xticks(x)
plt.xlabel('epoch')
plt.ylabel('score')
plt.title('Evaluation Scores')
plt.savefig(savepath + 'Evaluation Scores' + '.png')
