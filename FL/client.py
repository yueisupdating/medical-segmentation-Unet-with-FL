import numpy as np
import torch

from LossFunction import loss_function


class Client:
    def __init__(self, local_training_data, local_test_data, args, device):
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.args = args
        self.device = device
        self.criterion = loss_function  # .to(device)

    def train(self, net):
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        epoch_loss = []
        for epoch in range(self.args.epochs):
            total_loss = 0
            for images, labels in self.local_training_data:
                # images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                pred = net(images)
                loss = self.criterion(pred, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss.append(total_loss / len(self.local_training_data))
            self.global_test(net, self.local_test_data)
            print('Epoch Number: {} \t Loss: {:.6f}'.format(epoch, np.mean(epoch_loss)))
        return net, np.mean(epoch_loss)

    def global_test(self, model_global, global_test_data):
        model_global.eval()
        # model_global.to(self.device)
        test_loss = []
        with torch.no_grad():
            for img, tag in global_test_data:
                # input = input.to(self.device)
                # target = target.to(self.device)
                pred = model_global(img)
                loss = self.criterion(pred, tag)
                test_loss.append(loss.item())
            print("test loss:{}".format(np.mean(test_loss)))
