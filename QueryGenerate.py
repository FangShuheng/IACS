from torch_geometric.data import Data, Dataset, DataLoader
import networkx as nx
from multiprocessing import Pool
import random
import torch
import os.path
import numpy as np
import math
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class IACSQueryData(Data):
    def __init__(self, x, edge_index, y, query=None, pos = None, neg = None, edge_attr = None, bias = None, query_index = None):
        super(IACSQueryData, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.bias = bias
        self.y = y
        self.query = query
        self.query_index = query_index
        self.pos = pos
        self.neg = neg
        self.mask = torch.zeros(self.y.shape)
        self.mask[pos] = 1.0
        self.mask[neg] = 1.0

    def __inc__(self, key, value):
        if key == "query":
            return 0
        elif key == "query_index":
            return 1
        else:
            return super().__inc__(key, value)


class IACSAttributeQueryData(IACSQueryData):
    def __init__(self, x, edge_index, y, query=None, pos = None, neg = None, edge_attr = None, bias = None, query_index = None,
                 x_attr = None, attr_mask = None, attr_feats = None, raw_feats=None):
        super(IACSAttributeQueryData, self).__init__(x, edge_index, y, query, pos, neg, edge_attr, bias, query_index)
        self.x_attr = x_attr
        self.attr_mask = attr_mask
        self.attr_feats=attr_feats
        self.raw_feats=raw_feats

    def __inc__(self, key, value):
        if key == "x_attr":
            return 0
        else:
            return super().__inc__(key, value)

class TaskData(object):
    def __init__(self, all_queries_data, num_shots):
        self.all_queries_data  = all_queries_data
        self.num_shots = num_shots
        self.support_data, self.query_data = \
            self._support_query_split()
        self.num_support, self.num_query = len(self.support_data), len(self.query_data)

    def _support_query_split(self):
        random.shuffle(self.all_queries_data)
        support_data, query_data = self.all_queries_data[: self.num_shots], self.all_queries_data[self.num_shots:]
        return support_data, query_data

    def support_query_split(self):
        random.shuffle(self.all_queries_data)
        self.support_data, self.query_data = self.all_queries_data[: self.num_shots], self.all_queries_data[self.num_shots:]

    def get_batch(self):
        loader = DataLoader(self.all_queries_data, batch_size=len(self.all_queries_data), shuffle=False)
        return next(iter(loader))

    def get_support_batch(self):
        support_loader = DataLoader(self.support_data, batch_size=self.num_support, shuffle=True)  # already shuffled
        return next(iter(support_loader))

    def get_query_batch(self):
        query_loader = DataLoader(self.query_data, batch_size=self.num_query, shuffle=True)
        return next(iter(query_loader))



class RawGraphWithCommunity(object):
    def __init__(self, graph, communities, feats, embed_feats: None,
                 x_embed_feats:None,
                 min_community_size: int = 8,
                 attr_frequency: int = 3,
                 use_embed_feats=False):
        self.num_workers = 20
        self.graph = graph
        self.communities_pre = [community for community in communities if len(community) >= min_community_size]# list of list, filter invalid communities
        self.feats =feats
        self.x_feats = torch.from_numpy(self.feats)
        self.use_embed_feats = use_embed_feats
        self.embed_feats = embed_feats
        self.x_embed_feats = x_embed_feats
        self.query_index_pre = dict()
        self.query_index = dict()
        self.communities=dict()

        if self.use_embed_feats==True:
            num_attr = self.x_feats.size(-1)
            community_attr_freq = dict()
            for community_id, community in enumerate(self.communities_pre):
                for node in community:
                    if node not in self.query_index_pre:
                        self.query_index_pre[node] = set(community)
                    else:
                        self.query_index_pre[node] = self.query_index_pre[node].union(set(community))
                    # get the attribute
                    for attr in torch.nonzero(self.x_feats[node])[:, 0].tolist(): # for fast indexing/nonzero index
                        if community_id not in community_attr_freq:
                            community_attr_freq[community_id] = dict()
                        if attr not in community_attr_freq[community_id]:
                            community_attr_freq[community_id][attr] = 0
                        community_attr_freq[community_id][attr] += 1

            all_freq_attrs = set(range(num_attr))
            self.community_attribute_index = dict()
            # get the top 3 frequent attribute for each community as the candidate query attributes
            for community_id, attribute_freq in community_attr_freq.items():
                sorted_freq = sorted(attribute_freq.items(), key=lambda x: x[1], reverse=True) # sort the dict by value
                sorted_freq = sorted_freq[0 : min(len(sorted_freq), attr_frequency)] # top frequency list
                self.community_attribute_index[community_id] = set([attribute for (attribute, _) in sorted_freq])
                all_freq_attrs &= self.community_attribute_index[community_id]
            self.query_attributes = set()
            # remove the frequent attributes which appear in all the communities
            for community_id, freq_attr in self.community_attribute_index.items():
                self.community_attribute_index[community_id] = freq_attr.difference(all_freq_attrs)
                self.query_attributes |= self.community_attribute_index[community_id]
            for community_id, community in enumerate(self.communities_pre):
                if community_id in self.community_attribute_index.keys():
                    self.communities[community_id]=self.communities_pre[community_id]
                else:
                    for i,nodeid in enumerate(self.communities_pre[community_id]):
                        del self.query_index_pre[nodeid]
            for idx,node in enumerate(self.query_index_pre):
                self.query_index[node]=self.query_index_pre[node]
            self.num_communities=len(self.communities)
            print("num communities: {}".format(self.num_communities))
            self.query_attributes_index = {attr: idx for (idx, attr) in enumerate(self.query_attributes)} # remapping the attributes for querying
            self.num_query_attributes = len(self.query_attributes_index)
            self.number_of_queries=len(self.query_index)
            print("num query attribute: {}".format(self.num_query_attributes))
            print("num query: {}".format(len(self.query_index)))
        else:
            for community_id, community in enumerate(self.communities_pre):
                for node in community:
                    if node not in self.query_index_pre:
                        self.query_index_pre[node] = set(community)
                    else:
                        self.query_index_pre[node] = self.query_index_pre[node].union(set(community))
            self.community_attribute_index = dict()
            self.query_attributes = set()
            self.communities=self.communities_pre
            for idx,node in enumerate(self.query_index_pre):
                self.query_index[node]=self.query_index_pre[node]
            self.num_communities=len(self.communities)
            print("num communities: {}".format(self.num_communities))
            self.number_of_queries=len(self.query_index)
            print("num query: {}".format(len(self.query_index)))

        self.edge_index = torch.ones(size=(2, self.graph.number_of_edges()), dtype=torch.long)
        for i, e in enumerate(self.graph.edges()):
            self.edge_index[0][i], self.edge_index[1][i] = e[0], e[1]


    def sample_one_for_multi_node_attribute_query(self, query, num_pos, num_neg, max_query_node: int = 3, max_query_attribute: int = 1):
        num_query_node = random.randint(1, max_query_node)
        num_query_attribute = random.randint(1, max_query_attribute)
        pos = list(self.query_index[query])
        querys=random.sample(pos,k=num_query_node-1)
        querys.append(query)
        neg = list(set(range(self.graph.number_of_nodes())).difference(self.query_index[query]))
        pos = list(set(pos).difference(querys))
        if num_neg<=1:#ratio
            masked_pos = random.sample(pos, k = min(math.ceil(num_pos*(len(pos)+len(neg))), len(pos)))
            masked_neg = random.sample(neg, k = min(math.ceil(num_neg*(len(pos)+len(neg))), len(neg)))
        else:#number
            masked_pos = random.sample(pos, k = min(math.ceil(num_pos), len(pos)))
            masked_neg = random.sample(neg, k = min(math.ceil(num_neg), len(neg)))
        if len(self.community_attribute_index)==0:
            query_attributes=[]
        else:
            for community_id, community in enumerate(self.communities):
                if query in self.communities[community_id]:
                    query_attributes = random.sample(self.community_attribute_index[community_id], k=num_query_attribute)

        return querys, pos, neg, masked_pos, masked_neg, query_attributes

    def get_communities(self, task_size, num_shots):
        assert task_size <= len(self.query_index), "task size surpass the max number of queries."
        assert task_size > num_shots, "num shots surpasses the task size"
        queries = random.choices(list(self.query_index.keys()),k=task_size)
        return queries


    def get_one_IACS_attribute_query_tensor(self, query, num_pos, num_neg):
        query, pos, neg, masked_pos, masked_neg, query_attributes = self.sample_one_for_multi_node_attribute_query(query, num_pos, num_neg)
        x_q = torch.full((self.graph.number_of_nodes(), 1), -1)
        x_q[query] = 1
        x_q[masked_pos] = 1
        x_q[masked_neg]=0
        x = torch.cat([x_q, self.x_embed_feats], dim=-1) if self.use_embed_feats \
            else torch.cat([x_q, self.x_feats], dim=-1)
        x = x.to(torch.float32)

        # distance to query node, used as attention bias
        dist = {node: 0 for node in range(self.graph.number_of_nodes())}
        for q in query:
            q_dist = nx.single_source_shortest_path_length(self.graph, source=q)
            for node, val in q_dist.items():
                dist[node] += val
        dist = [dist[node] / float(len(query)) for node in
                range(self.graph.number_of_nodes())]  # average distance to query node
        dist = torch.FloatTensor(dist).unsqueeze(dim=-1) / max(dist)  # [num_nodes, 1], normalize the value

        y = torch.zeros(size=(self.graph.number_of_nodes(),), dtype=torch.float)
        y[pos] = 1

        query = torch.LongTensor(query)
        query_index = torch.zeros_like(query, dtype=torch.long)
        if len(query_attributes)==0:
             query_data = IACSAttributeQueryData(x=x, edge_index=self.edge_index, y=y, query=query,
                    pos=masked_pos, neg=masked_neg, bias= dist, query_index= query_index,raw_feats=self.x_feats)
        else:
            # assign the attribute features
            x_attr = torch.zeros(size=(self.graph.number_of_nodes(), self.num_query_attributes)).to(torch.float32) # [num_node, num_attribute]
            attr_mask = torch.zeros(size=(self.num_query_attributes,), dtype=torch.bool)
            for attribute in query_attributes:
                x_attr[:, self.query_attributes_index[attribute]] = self.x_feats[:, attribute] #[number of nodes,num query attributes]
                attr_mask[self.query_attributes_index[attribute]] = True
            attr_feats = torch.sum(self.embed_feats[query_attributes], dim=0, keepdim=False).to(torch.float32) if self.use_embed_feats else None
            query_data = IACSAttributeQueryData(x=x, edge_index=self.edge_index, y=y, query=query,
                                           pos=masked_pos, neg=masked_neg, bias= dist, query_index= query_index,
                                           x_attr=x_attr, attr_mask=attr_mask, attr_feats= attr_feats,raw_feats=self.x_feats)
        return query_data


    def get_attributed_task(self, queries, num_shots, meta_method, num_pos, num_neg):
        get_tensor_func = self.get_one_IACS_attribute_query_tensor
        all_queries_data = list()
        for query in queries:
            all_queries_data.append(get_tensor_func(query, num_pos, num_neg))
        task = TaskData(all_queries_data, num_shots=num_shots)
        return task