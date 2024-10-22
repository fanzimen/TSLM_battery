import torch.multiprocessing

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, visual_mse
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.anomaly_detection_metrics import adjbestf1
from utils.tools import adjustment
import collections
import csv
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from torchsummary import summary
import torch.nn.functional as F
class Exp_Anomaly_Detection_Mse(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_Mse, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                if self.args.use_ims:
                    # backward overlapping parts between outputs and inputs
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                    # input and output are completely aligned
                    outputs = self.model(batch_x, None, None, None)

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def finetune(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # summary(self.model, input_size=(1, 768, 1))
        print(self.model)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                if self.args.use_ims:
                    # backward overlapping parts between outputs and inputs
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                    # input and output are completely aligned
                    outputs = self.model(batch_x, None, None, None)


                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())


                # outputs = outputs
                # real, img = torch.chunk(outputs, dim=1, chunks=2)
                # outputs_freq = torch.complex(real, img)
                # outputs_temp = torch.fft.irfft(outputs_freq, dim=1)
                # outputs_temp = outputs_temp.permute(0, 2, 1)
                # # 使用插值将 outputs_temp 调整到 [1, 1, 768]
                # outputs_temp_resized = F.interpolate(outputs_temp, size=(768,), mode='linear', align_corners=False)
                # # 调整 outputs_temp_resized 的形状为 [1, 768, 1]
                # outputs_temp_resized = outputs_temp_resized.permute(0, 2, 1)
                # outputs_temp = outputs_temp_resized
                # batch_X = batch_x
                # loss_tmp = ((outputs-batch_x)**2).mean()
                # loss = criterion(outputs_temp, batch_X)
                # train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model

    def find_border(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border1_str = parts[-2]
        border2_str = parts[-1]
        if '.' in border2_str:
            border2_str = border2_str[:border2_str.find('.')]

        try:
            border1 = int(border1_str)
            border2 = int(border2_str)
            return border1, border2
        except ValueError:
            return None

    def find_border_number(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border_str = parts[-3]

        try:
            border = int(border_str)
            return border
        except ValueError:
            return None

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        score_list = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        border_start = self.find_border_number(self.args.data_path)
        border1, border2 = self.find_border(self.args.data_path)

        # test_label = np.zeros(test_data.data.shape[0])
        # test_label[border1 - border_start:border2 - border_start] = 1

        token_count = 0
        if self.args.use_ims:
            rec_token_count = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len
        else:
            rec_token_count = self.args.seq_len // self.args.patch_len

        input_list = []
        output_list = []
        score_list = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device) #[1, 768, 1]
                # reconstruct the input sequence and record the loss as a sorted list
                if self.args.use_ims:
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None) #[1, 672, 1] -> [1, 672, 1]
                    batch_x = batch_x[:, self.args.patch_len:-self.args.patch_len, :] #[1, 576, 1]
                    outputs = outputs[:, :-self.args.patch_len, :] #[1, 576, 1]
                else: 
                    outputs = self.model(batch_x, None, None, None)


                # real, img = torch.chunk(outputs, dim=1, chunks=2)
                # outputs_freq = torch.complex(real, img)
                # outputs_temp = torch.fft.irfft(outputs_freq, dim=1)
                # # 调整 outputs_temp 的形状为 [1, 1, 766]
                # outputs_temp = outputs_temp.permute(0, 2, 1)
                # # 使用插值将 outputs_temp 调整到 [1, 1, 768]
                # outputs_temp_resized = F.interpolate(outputs_temp, size=(768,), mode='linear', align_corners=False)
                # # 调整 outputs_temp_resized 的形状为 [1, 768, 1]
                # outputs_temp_resized = outputs_temp_resized.permute(0, 2, 1)
                # outputs = outputs_temp_resized

                input_list.append(batch_x[0, -self.args.patch_len:, -1].detach().cpu().numpy())
                output_list.append(outputs[0, -self.args.patch_len:, -1].detach().cpu().numpy())
                test_labels.append(batch_y[0,-self.args.patch_len:].reshape(-1).detach().cpu().numpy())
                score = self.anomaly_criterion(batch_x, outputs).reshape(-1) 
                score_list.append(score[-self.args.patch_len:].detach().cpu().numpy()) #(1, 96)






        test_labels = np.concatenate(test_labels, axis=0).flatten()
        input = np.concatenate(input_list, axis=0).reshape(-1)
        output = np.concatenate(output_list, axis=0).reshape(-1)
        score_list = np.concatenate(score_list, axis=0).reshape(-1)

        best_adjusted_f1, best_pred = adjbestf1(test_labels, score_list, 100)
        print('best_adjusted_f1:', best_adjusted_f1)

        gt = test_labels.astype(int)
        accuracy = accuracy_score(gt, best_pred.astype(int))
        precision, recall, f_score, support = precision_recall_fscore_support(gt, best_pred.astype(int), average='binary')
        gt, adj_pred = adjustment(gt, best_pred)
        adjaccuracy = accuracy_score(gt, adj_pred)
        adjprecision, adjrecall, adjf_score, adjsupport = precision_recall_fscore_support(gt, adj_pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        print("adjAccuracy : {:0.4f}, adjPrecision : {:0.4f}, adjRecall : {:0.4f}, adjF-score : {:0.4f} ".format(
            adjaccuracy, adjprecision,adjrecall, adjf_score))




        data_path = os.path.join(folder_path, self.args.data_path[:self.args.data_path.find('.')])
        file_path = data_path + '.png'
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        # border1 = border1 - 768
        # border2 = border2 - 768
        visual_mse(input[border1 - border_start-300:border2 - border_start+300], output[border1 - border_start-300:border2 - border_start+300], 
                   score_list[border1 - border_start-300:border2 - border_start+300], test_labels[border1 - border_start-300:border2 - border_start+300],file_path)
        if border1 - border_start - 1200 < 0:
            file_path1 = data_path + '_after.png'
            visual_mse(input[border2 - border_start+300:border2 - border_start+1200], output[border2 - border_start+300:border2 - border_start+1200], 
                   score_list[border2 - border_start+300:border2 - border_start+1200], test_labels[border2 - border_start+300:border2 - border_start+1200],file_path1)
        else:
            file_path1 = data_path + '_before.png'
            visual_mse(input[border1 - border_start - 1200:border1 - border_start - 300], output[border1 - border_start - 1200:border1 - border_start - 300], 
                   score_list[border1 - border_start - 1200:border1 - border_start - 300], test_labels[border1 - border_start - 1200:border1 - border_start - 300],file_path1)

        accuracy = accuracy.astype(np.float32)
        recall = recall.astype(np.float32)
        f_score = f_score.astype(np.float32)
        adjaccuracy = adjaccuracy.astype(np.float32)
        adjprecision = adjprecision.astype(np.float32)
        adjrecall = adjrecall.astype(np.float32)
        adjf_score = adjf_score.astype(np.float32)
        # Write results to CSV file
        results = {
            "setting": [setting],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F-score': f_score,
            'adjAccuracy': adjaccuracy,
            'adjPrecision': adjprecision,
            'adjRecall': adjrecall,
            'adjF-score': adjf_score
        }
        # 将非迭代的值包装在列表中
        for key in results:
            if not isinstance(results[key], collections.abc.Iterable):
                results[key] = [results[key]]

        csv_file = 'results.csv'


        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(results.keys())
            writer.writerows(zip(*results.values()))

        print("Results appended to", csv_file)
        return
