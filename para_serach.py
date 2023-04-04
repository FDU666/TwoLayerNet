import numpy as np
from tqdm import tqdm
import model1
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

f = open('./para_search.txt', 'w')

learning_rate_list = [0.001, 0.005, 0.01]
hidden_dim_list = [100, 500, 1000]
regularization_list = [0, 0.0001, 0.001]

best_lr = 0
best_hidden_dim = 0
best_reg = 0
best_acc = 0

train_images, train_labels = model1.load_mnist('./minist')
test_images, test_labels = model1.load_mnist('./minist', 't10k')

for i in tqdm(range(len(learning_rate_list))):
    for j in range(len(hidden_dim_list)):
        for k in range(len(regularization_list)):
            print(i, j, k)
            np.random.seed(123)

            lr = learning_rate_list[i]
            hidden_dim = hidden_dim_list[j]
            reg = regularization_list[k]

            twolayermodel = model1.Two_Layer_Net(train_images.shape[1], hidden_dim, 10)
            train_loss_history, test_loss_history, test_acc_history = twolayermodel.train(train_images, train_labels,
                                                                                          test_images, test_labels,
                                                                                          learning_rate=lr,
                                                                                          regulariaztion=reg)
            if max(test_acc_history) > best_acc:
                best_acc = max(test_acc_history)
                best_lr = lr
                best_hidden_dim = hidden_dim
                best_reg = reg
                twolayermodel.save_model('./bestmodel.npz')

            f.write('learning_rate:')
            f.write(' ')
            f.write(str(learning_rate_list[i]))
            f.write(' ')
            f.write('hidden_dim:')
            f.write(' ')
            f.write(str(hidden_dim_list[j]))
            f.write(' ')
            f.write('regularization:')
            f.write(' ')
            f.write(str(regularization_list[k]))
            f.write(' ')
            f.write('Accuracy:')
            f.write(' ')
            f.write(str(max(test_acc_history)))
            f.write('\n')

f.write('Best learning_rate:')
f.write(' ')
f.write(str(best_lr))
f.write(' ')
f.write('Best hidden_dim:')
f.write(' ')
f.write(str(best_hidden_dim))
f.write(' ')
f.write('Best regularization:')
f.write(' ')
f.write(str(best_reg))
f.write(' ')
f.write('Best Accuracy:')
f.write(' ')
f.write(str(best_acc))
f.write('\n')
f.close()