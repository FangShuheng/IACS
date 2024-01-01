from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from data_load import load_twitter_graphs, load_arxiv, seed_all
from train_eval import IACS
from Model import CSIACSComp
import torch.optim as optim
import torch
import numpy as np
import wandb
from tqdm import tqdm
from tqdm.contrib import tzip
import os
import time
import networkx as nx
from torch.utils.data import DataLoader
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)


def main(args):
    seed_all(args.seed)
    wandb_run=wandb.init(config=args,project='IACS',dir='/home/shfang/IACS/wandb/',job_type="training",name="fe01{}_{}_train{}_valid{}_test{}_pos{}_neg{}".format(args.data_set,args.meta_method,args.train_task_num,args.valid_task_num,args.test_task_num,args.num_pos,args.num_neg),reinit=True)
    task_size = args.task_size
    num_shots = args.num_shots
    num_pos, num_neg = args.num_pos, args.num_neg

    if args.data_set=='twitter':
        #attribute community search
        raw_data_list = load_twitter_graphs(args, "/home/shfang/data/twitter/twitter",use_embed_feats=args.use_embed_feats)
        node_feat= 128
        raw_data_list = [raw_data for raw_data in raw_data_list # filter invalid raw data
                     if raw_data.num_communities > 1 and raw_data.num_query_attributes > 0]
        communities_list = [raw_data.get_communities(task_size, num_shots) for raw_data in raw_data_list]
        tasks = [raw_data.get_attributed_task(community_ids, num_shots, args.meta_method, num_pos, num_neg)
                for raw_data, community_ids in zip(raw_data_list, communities_list)]
        tn=len(tasks)
        print("len:{}".format(tn))
        train_tasks, valid_tasks, test_tasks = tasks[0:int(tn*0.7)], tasks[int(tn*0.7):int(tn*0.8)], tasks[int(tn*0.8):]


    elif args.data_set=='arxiv':
        #Non-attribute community search
        raw_data_list_train, raw_data_list_valid, raw_data_list_test, node_feat = load_arxiv(args)
        print('get queries!')
        queries_list_train = [raw_data.get_communities(task_size, num_shots) for raw_data in raw_data_list_train]
        queries_list_valid = [raw_data.get_communities(task_size, num_shots) for raw_data in raw_data_list_valid]
        queries_list_test = [raw_data.get_communities(task_size, num_shots) for raw_data in raw_data_list_test]
        print('get tasks!')
        train_tasks = [raw_data.get_attributed_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
                   tzip(raw_data_list_train, queries_list_train)]
        valid_tasks=[raw_data.get_attributed_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
                  tzip(raw_data_list_valid, queries_list_valid)]
        test_tasks = [raw_data.get_attributed_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
                  tzip(raw_data_list_test, queries_list_test)]



    if args.meta_method in ["IACS", "iacs"]:
        if args.use_embed_feats==False:
            print('----------IACS for Non-attribute----------')
            model = CSIACSComp(args, node_feat_dim=node_feat + 1, edge_feat_dim=10, decoder_type=args.decoder_type)
            print(model)
            IACS = IACS(args, model, wandb_run)
            print('begin training...')
            t=IACS.train_IACS(train_tasks,valid_tasks,test_tasks)
            print('begin test!')
            acc, precision, recall, f1, acc_com, precision_com, recall_com, f1_com,t2=IACS.evaluate_IACS(test_tasks,args.epochs)
            print('train_time={:.4f}, test_time={:.4f}'.format(t,t2))
        else:
            print('----------IACS for Attribute CS----------')
            model = CSIACSComp(args, node_feat_dim=node_feat + 1, edge_feat_dim=10, decoder_type=args.decoder_type)
            print(model)
            IACS = IACS(args, model, wandb_run)
            print('begin training...')
            t=IACS.train_IACS(train_tasks,valid_tasks,test_tasks)
            print('begin test!')
            acc, precision, recall, f1, acc_com, precision_com, recall_com, f1_com,t2=IACS.evaluate_IACS(test_tasks,args.epochs)
            print('train_time={:.4f}, test_time={:.4f}'.format(t,t2))


if __name__ == "__main__":
    parser = ArgumentParser("IACS", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--num_layers", default=3, type=int,
                        help="number of gnn conv layers")
    parser.add_argument("--gnn_type", default="GAT", type=str, # GCN, GAT, GATBias
                        help="GNN type")
    parser.add_argument("--pool_type", default="avg", type=str,  # att, sum, avg
                        help="IACS Context Pool Type")
    parser.add_argument("--decoder_type", default="IP", type=str,
                        help="IACS Decoder Type")
    parser.add_argument("--film_type", default="no", type=str,  # gate, no, plain
                        help="Context FiLM Layer Type")
    parser.add_argument("--gnn_act_type", default="relu", type=str,
                        help="activation layer inside gnn aggregate/combine function")
    parser.add_argument("--act_type", default="relu", type=str,
                        help="activation layer function for MLP and between GNN layers")
    parser.add_argument("--embed_type", default="prone", type=str,
                        help="the node feature encoding type")
    parser.add_argument("--num_g_hid", default=128, type=int,
                        help="hidden dim for transforming nodes")
    parser.add_argument("--num_e_hid", default=128, type=int,
                        help="hidden dim for transforming edges")
    parser.add_argument("--gnn_out_dim", default=128, type=int,
                        help="number of output dimension")
    parser.add_argument("--mlp_hid_dim", default=512, type=int,
                        help="number of hidden units of MLP")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--batch_norm", default=False, type=bool)
    parser.add_argument("--use_embed_feats", action='store_true', default=True, help="input use the embed features")

    #Settings
    parser.add_argument("--meta_method", default="IACS", type=str,
                        help="The meta learning algorithm")
    parser.add_argument("--task_size", default=24, type=int)
    parser.add_argument("--num_shots", default=8, type=int)
    parser.add_argument("--num_pos", default=0.05, type=float)
    parser.add_argument("--num_neg", default=0.05, type=float)
    parser.add_argument("--train_task_num", type=int, help='the number of training task', default=128)
    parser.add_argument("--valid_task_num", type=int, help='the number of training task', default=32)
    parser.add_argument("--test_task_num", type=int, default=32, help='the number of test task')
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--finetune_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default= 1e-5, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--scheduler_type", default="exponential", type=str,
                        help="the node feature encoding type")
    parser.add_argument('--decay_factor', type=float, default=0.8,
                        help='decay rate of (gamma).')
    parser.add_argument('--decay_patience', type=int, default=10,
                        help='num of epochs for one lr decay.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--num_workers', type = int, default= 16,
                        help='number of workers for Dataset.')

    # Input/Output dir
    parser.add_argument("--data_dir", type=str, default="/home/shfang/data/facebook/facebook")
    parser.add_argument("--project_dir", type=str, default="/home/shfang/IACS/")
    parser.add_argument("--data_set", default='twitter', type=str, help='dataset')
    parser.add_argument("--verbose", default=True, type=bool)
    parser.add_argument("--Test", default=False, action='store_true')
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    # set the hardware parameter
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    if args.verbose:
        print(args)
    main(args)

