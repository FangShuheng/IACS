import os
import networkx as nx
import numpy as np
import random
import torch
from QueryGenerate import RawGraphWithCommunity
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
import torch_geometric
import nxmetis
from tqdm import tqdm
import time
from ogb.nodeproppred import PygNodePropPredDataset



def seed_all(seed: int =0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("set all seed!")

def evaluate_prediction(pred, targets):
    acc = accuracy_score(targets, pred)
    precision = precision_score(targets, pred)
    recall = recall_score(targets, pred)
    f1 = f1_score(targets, pred)
    return acc, precision, recall, f1

def np_save_if_not_existed(path, saved_data):
    if not os.path.exists(path):
        saved_data_numpy = np.array([saved_data], dtype=object)
        np.save(path, saved_data_numpy)


def load_twitter_graphs(args, data_dir: str,return_attr_info = False, use_embed_feats = False):
    raw_data_list = list()
    attr_info_list=list()
    for file_name in os.listdir(data_dir):
        if os.path.splitext(file_name)[1] != '.edges':
            continue
        ego_node_id = int(os.path.splitext(file_name)[0])
        feat_dict = dict()
        with open(os.path.join(data_dir, "{}.featnames".format(ego_node_id)), 'r') as feat_names:
            for line in feat_names:
                tokens = line.strip().split()
                f_id = int(tokens[0])
                feat_name = '+'.join(tokens[1:])
                feat_dict[f_id] = feat_name
            feat_names.close()

        # load input feats
        node_id_dict = dict()
        node_attrs_dict=dict()
        node_cnt = 0
        with open(os.path.join(data_dir, "{}.feat".format(ego_node_id)), 'r') as feat:
            lines = feat.readlines()
            feats = np.zeros(shape=(len(lines) + 1, len(feat_dict)), dtype=float) # why +1? add ego node
            for line in lines:
                tokens = line.strip().split()
                node_attrs_dict[node_cnt]=list()
                for i, val in enumerate(tokens[1:]):
                    if int(val) <= 0:
                        continue
                    idx = i
                    feats[node_cnt][idx] = 1
                    node_attrs_dict[node_cnt].append(str(i))
                node_id_dict[int(tokens[0])] = node_cnt
                node_cnt += 1
            feat.close()
        with open(os.path.join(data_dir, "{}.egofeat".format(ego_node_id)), 'r') as egofeat:
            node_id_dict[ego_node_id] = node_cnt
            node_attrs_dict[node_cnt]=list()
            for line in egofeat:
                tokens = line.strip().split()
                for i, val in enumerate(tokens):
                    if int(val) <= 0:
                        continue
                    idx = i
                    feats[node_cnt][idx] = 1
                    node_attrs_dict[node_cnt].append(str(i))
            egofeat.close()

        # load graph edges:
        edge_list = list()

        with open(os.path.join(data_dir, "{}.edges".format(ego_node_id)), "r") as edges:
            for line in edges:
                tokens = line.strip().split()
                src, dst = int(tokens[0]), int(tokens[1])
                edge_list.append((node_id_dict[src], node_id_dict[dst]))
            edges.close()

            ego_edges = [(node_id_dict[ego_node_id], k) for k in node_id_dict.values()]
            edge_list += ego_edges

        # load communities info
        communities = list()
        with open(os.path.join(data_dir, "{}.circles".format(ego_node_id)), 'r') as circles:
            for line in circles:
                tokens = line.strip().split()
                node_ids = [node_id_dict[int(token)] for token in tokens[1:]]
                communities.append(node_ids)
            circles.close()
        if len(communities)<1:
            continue

        graph = nx.Graph()
        graph.add_edges_from(edge_list)
        if not nx.is_connected(graph):
            print('skip')
            continue
        print("# of nodes/edges:", graph.number_of_nodes(), graph.number_of_edges(),"id:",ego_node_id)

        embed_feats_path=data_dir+"/emb/egotwitter_enhanced{}.emb.npy".format(ego_node_id)

        embed_feats = torch.from_numpy(np.load(embed_feats_path,allow_pickle=True))
        x_embed_feats = torch.zeros(size=(graph.number_of_nodes(), embed_feats.size(-1)), dtype=torch.float)
        for _,node in enumerate(node_id_dict.keys()):
            attrs_idx = torch.nonzero(torch.from_numpy(feats[node_id_dict[node]]))[:, 0].tolist()
            if len(attrs_idx) <= 0:
                continue
            x_embed_feats[node_id_dict[node]] = torch.mean(embed_feats[attrs_idx], dim=0, keepdim=False)
        raw_data = RawGraphWithCommunity(graph, communities, feats, embed_feats, x_embed_feats,use_embed_feats=use_embed_feats, min_community_size=16)
        if raw_data.number_of_queries<args.task_size:
            print('skip')
            continue
        raw_data_list.append(raw_data)
        attr_info_list.append((node_attrs_dict, edge_list))
    print(len(raw_data_list))
    if return_attr_info:
        return raw_data_list,  attr_info_list
    return raw_data_list

def load_arxiv(args):
    training_raw_data_list =  list()
    valid_raw_data_list = list()
    test_raw_data_list=list()
    if os.path.exists(os.path.join(args.project_dir, 'saved_subgraph_arxiv'))  and os.listdir(os.path.join(args.project_dir, 'saved_subgraph_arxiv')): #if path exists and is not none
        for file_name in os.listdir(os.path.join(args.project_dir, 'saved_subgraph_arxiv')):
            idx=int(file_name.strip().split('_')[-1].split('.')[0])
            print(f"subgraph of {idx} found, loading from file")
            raw_data_path = os.path.join(args.project_dir, 'saved_subgraph_arxiv', file_name)
            raw_data = np.load(raw_data_path, allow_pickle=True)[0]
            if raw_data.number_of_queries<args.task_size:
                print('skip')
                continue
            if idx % 10 ==0 or idx % 10==1:
                test_raw_data_list.append(raw_data)
            elif idx % 10 ==2:
                valid_raw_data_list.append(raw_data)
            else:
                training_raw_data_list.append(raw_data)
            print("subgraph of {} size: {}, {},length of communities {},feats size({},{})".format(idx,raw_data.graph.number_of_nodes(), raw_data.graph.number_of_edges(),len(raw_data.communities),raw_data.feats.shape[0],raw_data.feats.shape[1]))
        num_feat=raw_data.feats.shape[1]
        return training_raw_data_list, valid_raw_data_list, test_raw_data_list, num_feat
    else: #if path not exists and is none
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/shfang/IACS/data/arxiv/')
        data=dataset[0]
        graph=torch_geometric.utils.to_networkx(data,to_undirected=True)
        print(graph.number_of_nodes(), graph.number_of_edges())

        glob_communities = dict()
        for node_id, label in enumerate(data.y.numpy().tolist()):
            label=int(label[0])
            if label not in glob_communities.keys():
                glob_communities[label] = list()
            glob_communities[label].append(node_id)

        if not os.path.exists(os.path.join(args.project_dir, 'saved_subgraph_arxiv')):
            os.makedirs(os.path.join(args.project_dir, 'saved_subgraph_arxiv'))
        print(graph.number_of_nodes(), graph.number_of_edges())
        print("begin partition---")
        time_begin=time.time()
        obj,subgraph_nodes=nxmetis.partition(graph,args.train_task_num+args.valid_task_num+args.test_task_num)
        time_end=time.time()
        print("end partition---")
        print("time cost:{}".format(time_end-time_begin))

        #generate training/valid/test task
        for idx in tqdm(range(len(subgraph_nodes))):
            raw_data_name = f"arxiv_subgraph_of_{idx}.npy"
            raw_data_path = os.path.join(args.project_dir, 'saved_subgraph_arxiv', raw_data_name)
            node_list=subgraph_nodes[idx]

            edge_list=graph.subgraph(node_list).edges()
            old_graph=nx.Graph()
            old_graph.add_edges_from(edge_list)
            if not nx.is_connected(old_graph):
                print('skip')
                continue
            node_id_dict = {l: n_id for n_id, l in enumerate(old_graph.nodes)}#key original id; value new id;
            node_new2old = {n_id:l for n_id, l in enumerate(old_graph.nodes)}
            res_graph=nx.Graph()
            edge_list = [(node_id_dict[src], node_id_dict[dst]) for (src, dst) in edge_list]
            res_graph.add_edges_from(edge_list)

            node_list=list(res_graph.nodes())
            node_list=[node_new2old[l] for n_id,l in enumerate(res_graph.nodes)]
            feats=data.x[node_list].numpy()
            #import pdb;pdb.set_trace()

            communities=list()
            candidate_query_number=0
            for k, val in glob_communities.items():#key:label value:node id
                temp_comm = set(val).intersection(set(node_list))  # get the local community induced by node_list
                temp_comm = [node_id_dict[node] for node in temp_comm]
                communities.append(temp_comm)
                candidate_query_number=candidate_query_number+len(temp_comm)
            if len(communities)<2:
                continue
            embed_feats=[]
            x_embed_feats=[]
            raw_data=RawGraphWithCommunity(res_graph, communities, feats, embed_feats, x_embed_feats)
            np_save_if_not_existed(raw_data_path, raw_data)

            if raw_data.number_of_queries<args.task_size:
                print('skip')
                continue

            if idx % 10 ==0 or idx % 10==1:
                test_raw_data_list.append(raw_data)
            elif idx % 10 ==2:
                valid_raw_data_list.append(raw_data)
            else:
                training_raw_data_list.append(raw_data)
            print("subgraph of {} size: {}, {},length of communities {},feats size ({},{})".format(idx,res_graph.number_of_nodes(), res_graph.number_of_edges(),len(communities),feats.shape[0],feats.shape[1]))
        num_feat=feats.shape[1]
        print(len(training_raw_data_list),len(valid_raw_data_list),len(test_raw_data_list))
        return training_raw_data_list, valid_raw_data_list, test_raw_data_list, num_feat




