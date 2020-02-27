import torch
import os
import cv2
import torch.utils.data as Data
import numpy as np
from matplotlib import pyplot as plt
from mask_detect_cnn_model import CNN





def load_pic(path=('cropped_mask', 'cropped_nomask')):
    x = []
    y = []
    file_names = os.listdir(path[0])
    for file_name in file_names:
        pic = cv2.imread(path[0] + '/' + file_name).astype('float32')
        x.append(pic.T.copy())
        y.append(1)
    file_names = os.listdir(path[1])
    for file_name in file_names:
        pic = cv2.imread(path[1] + '/' + file_name)
        x.append(pic.T.copy())
        y.append(0)
    print('Pics loaded')
    return torch.tensor(x), torch.tensor(y)


def save_net(network):
    torch.save(network, 'mask_detect_cnn.pkl')


def save_net_parameters(network):
    torch.save(network.state_dict(), 'mask_detect_cnn_parameters.pkl')


if __name__ == '__main__':
    BATCH_SIZE = 5
    lossing = []
    data_tensor, target_tensor = load_pic()
    data_set = Data.TensorDataset(data_tensor, target_tensor)
    loader = Data.DataLoader(dataset=data_set, batch_size=BATCH_SIZE, shuffle=True)

    cnn_net = CNN()
    optimiser = torch.optim.Adam(cnn_net.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(20):
        for batch_num, (pic_data, target_data) in enumerate(loader):
            prediction = cnn_net(pic_data)
            loss = loss_func(prediction, target_data)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print('Epoch:%d|Batch:%d|Loss%f' % (epoch, batch_num, loss.detach().numpy()))
            lossing.append(loss.detach().numpy())
    plt.plot([i for i in range(len(lossing))], lossing)
    plt.show()
    save_net(cnn_net)
    save_net_parameters(cnn_net)

    test_data_tensor, test_target_tensor = load_pic(('result1', 'result2'))
    test_data_set = Data.TensorDataset(test_data_tensor, test_target_tensor)
    loader = Data.DataLoader(dataset=test_data_set, batch_size=1, shuffle=True)
    yes, no = 0, 0
    for pic_data, target_data in zip(test_data_tensor, test_target_tensor):
        prediction = cnn_net(pic_data.unsqueeze(0)).detach().numpy()
        label_predicted = prediction.tolist()[0].index(prediction[0].max())
        label_real = target_data.numpy()
        if label_predicted == label_real:
            yes += 1
            print(label_predicted, '---', label_real, 'Y')
        else:
            no += 1
            print(label_predicted, '---', label_real, 'N')
    print(yes/(yes+no))
