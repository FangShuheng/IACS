import torch
import torch.nn as nn
from torch import optim
from data_load import evaluate_prediction
import random
import numpy as np
import torch.nn.functional as F
from Loss import WeightBCEWithLogitsLoss
import os
from tqdm import tqdm
import copy
import time


class IACS(nn.Module):
    def __init__(self, args, model,wandb_run):
        super(IACS, self).__init__()
        self.args = args
        self.model = model
        self.wandb_run=wandb_run
        self.wandb_run.watch(self.model, log="all")
        self.learning_rate = args.learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=args.decay_factor, patience=args.decay_patience)
        self.criterion = WeightBCEWithLogitsLoss()
        if self.args.cuda:
            self.model.to(self.args.device)

    def train_IACS(self, train_tasks, valid_tasks, test_tasks):
        self.wandb_run.watch(self.model,log='all')
        self.model.train()
        for epoch in tqdm(range(self.args.epochs)):
            for task in train_tasks:
                task.support_query_split()
            support_batches = [ task.get_support_batch() for task in train_tasks]
            query_batches = [ task.get_batch() for task in train_tasks]
            print('***begin***')
            self.valid_IACS(valid_tasks,epoch)
            qry_loss = 0.0
            all_preds = list()
            all_targets = list()
            for i in range(len(support_batches)):
                self.optimizer.zero_grad()
                support_batch = support_batches[i]
                query_batch = query_batches[i]
                if self.args.cuda:
                    support_batch = support_batch.to(self.args.device)
                    query_batch = query_batch.to(self.args.device)
                t_begin=time.time()
                output, y, mask = self.model(support_batch, query_batch,'qry')
                loss = self.criterion(output, y, mask)
                loss.backward()
                qry_loss += loss.item()/len(support_batches)
                t_end=time.time()
                t=t+t_end-t_begin
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step(qry_loss)
            print("{}-th Epoch: BCE Loss={:.4f}".format(epoch, qry_loss))
            print("**end**")
            self.wandb_run.log({'training_qry_loss': qry_loss},step=epoch)
        print('Train_time={} for {} epochs'.format(t,self.args.epochs))
        return t

    def evaluate_IACS(self, test_tasks,epoch):
        self.model.eval()
        i = 0
        all_preds = list()
        all_targets = list()
        qry_loss=0
        support_batches = [ task.get_support_batch() for task in test_tasks]
        query_batches = [ task.get_query_batch() for task in test_tasks]
        t=0
        tt=0
        for i in range(len(support_batches)):
            support_batch = support_batches[i]
            query_batch = query_batches[i]
            if self.args.cuda:
                support_batch = support_batch.to(self.args.device)
                query_batch = query_batch.to(self.args.device)
            model=copy.deepcopy(self.model)
            model.load_state_dict(self.model.state_dict())
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
            t_begin=time.time()
            for ep in range(self.args.finetune_epochs):
                optimizer.zero_grad()
                output, y, mask = model(support_batch, query_batch,'spt')
                loss = self.criterion(output, y, mask)
                loss.backward()
                optimizer.step()
            t_mid=time.time()
            t=t+t_mid-t_begin
            output, y, mask = model(support_batch, query_batch,'qry')
            t_end=time.time()
            tt=tt+t_end-t_mid
            loss = self.criterion(output, y, mask)
            qry_loss += loss.item()/len(support_batches)

            pred = torch.sigmoid(output)
            pred = torch.where(pred > 0.5, 1.0, 0.0)

            pred, targets = pred.view(-1), y.view(-1)
            pred, targets = pred.cpu().detach().numpy(), targets.cpu().detach().numpy()

            all_preds.append(pred)
            all_targets.append(targets)
            acc, precision, recall, f1 = evaluate_prediction(pred, targets)

            print("IACS Test Task-{}: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format(i, acc, precision, recall, f1))
            i += 1

        all_preds = np.hstack(all_preds)
        all_targets = np.hstack(all_targets)
        acc, precision, recall, f1 = evaluate_prediction(all_preds, all_targets)

        self.wandb_run.log({'test_qry_loss': qry_loss},step=epoch)
        print("Epoch {} IACS Test Res: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format(epoch, acc, precision, recall, f1))
        print("Epoch {} finetune_time={}, IACS_Test_time= {}".format(epoch, t,tt))
        return acc, precision, recall, f1, tt


    def valid_IACS(self, valid_tasks,epoch):
        self.model.eval()
        i = 0
        t=0
        all_preds = list()
        all_targets = list()
        qry_loss=0
        support_batches = [ task.get_support_batch() for task in valid_tasks]
        query_batches = [ task.get_query_batch() for task in valid_tasks]
        for i in range(len(support_batches)):
            support_batch = support_batches[i]
            query_batch = query_batches[i]
            if self.args.cuda:
                support_batch = support_batch.to(self.args.device)
                query_batch = query_batch.to(self.args.device)
            t_begin=time.time()
            output, y, mask = self.model(support_batch, query_batch,'qry')
            t_end=time.time()
            t=t+t_end-t_begin
            loss = self.criterion(output, y)
            qry_loss += loss.item()/len(support_batches)
            pred = torch.sigmoid(output)
            pred = torch.where(pred > 0.5, 1, 0)
            pred, targets = pred.view(-1), y.view(-1)
            pred, targets = pred.cpu().detach().numpy(), targets.cpu().detach().numpy()
            all_preds.append(pred)
            all_targets.append(targets)
            acc, precision, recall, f1 = evaluate_prediction(pred, targets)
            i += 1
        all_preds = np.hstack(all_preds)
        all_targets = np.hstack(all_targets)
        acc, precision, recall, f1 = evaluate_prediction(all_preds, all_targets)
        self.wandb_run.log({'val_acc': acc},step=epoch)
        self.wandb_run.log({'val_pre': precision},step=epoch)
        self.wandb_run.log({'val_recall': recall},step=epoch)
        self.wandb_run.log({'val_f1': f1},step=epoch)
        self.wandb_run.log({'val_qry_loss': qry_loss},step=epoch)
        print("Epoch {} Valid Res: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format(epoch,acc, precision, recall, f1))
        print('valid time:{}'.format(t))


