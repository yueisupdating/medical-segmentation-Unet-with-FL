import copy
import numpy as np
import torch

from client import Client


class FedAvgTrainer(object):
    def __init__(self, train_data_local_dict, test_data_local_dict, model, device, args):
        self.client_list = []
        self.device = device
        self.args = args
        self.model_global = model
        self.model_global.train()
        self.setup_clients(train_data_local_dict, test_data_local_dict)

    def setup_clients(self, train_data_local_dict, test_data_local_dict):
        print("############setup_clients (START)#############")
        for client_idx in range(self.args.client_number):
            c = Client(train_data_local_dict[client_idx], test_data_local_dict[client_idx], args=self.args,
                       device=self.device)
            self.client_list.append(c)
        print("############setup_clients (END)#############")

    def train(self):
        for round_idx in range(self.args.comm_round):
            print("Communication round : {}".format(round_idx))
            self.model_global.train()
            w_locals, loss_locals = [], []
            for idx, client in enumerate(self.client_list):
                w, loss = client.train(net=copy.deepcopy(self.model_global))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            w_glob = self.aggregate(w_locals)
            self.model_global.load_state_dict(w_glob)
            loss_avg = np.mean(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(round_idx, loss_avg))
            torch.save(self.model_global.state_dict(), 'my_model.pth')

    def aggregate(self, w_locals):
        print("################aggregate: %d" % len(w_locals))
        averaged_params = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_model_params = w_locals[i]
                w = 1 / len(self.client_list)
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
