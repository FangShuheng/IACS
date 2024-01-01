import torch.nn as nn
import torch
import torch.nn.functional as F
from Layer import MLP, FiLM, GATBiasConv
from Layer import get_act_layer
from torch_scatter import scatter_mean
from torch_geometric.nn import GATConv, GraphConv, SAGEConv



class GNN(nn.Module):
    def __init__(self, args, node_feat_dim, edge_feat_dim):
        super(GNN, self).__init__()
        self.num_node_feat = node_feat_dim
        self.num_edge_feat = edge_feat_dim
        self.num_layers = args.num_layers
        self.num_hid = args.num_g_hid
        self.num_e_hid = args.num_e_hid
        self.num_out = args.gnn_out_dim
        self.model_type = args.gnn_type
        self.dropout = args.dropout
        self.convs = nn.ModuleList()
        self.act_type = args.act_type
        self.act_layer = get_act_layer(self.act_type)
        self.gnn_act_layer = get_act_layer(args.gnn_act_type)
        cov_layer = self.build_cov_layer(self.model_type)

        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid

            if self.model_type == "GAT" or self.model_type == "GCN" or self.model_type == "SAGE" or self.model_type == "GATBias":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
            else:
                raise NotImplementedError("Unsupported model type!")


    def build_cov_layer(self, model_type):
        if model_type == "GAT":
            return GATConv
        elif model_type == "GATBias":
            return GATBiasConv
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "GCN":
            return GraphConv
        else:
            raise NotImplementedError("Unsupported model type!")

    def forward(self, x, edge_index, x_batch, edge_attr = None, att_bias = None):
        for i in range(self.num_layers):
            if self.model_type == "GAT" or self.model_type =="GCN" or self.model_type == "SAGE":
                x = self.convs[i](x, edge_index)
            elif self.model_type == "GATBias":
                x = self.convs[i](x, edge_index, att_bias)
            else:
                print("Unsupported model type!")

            if i < self.num_layers - 1:
                if self.act_type != 'relu':
                    x = self.act_layer(x)
                x = F.dropout(x, p = self.dropout, training=self.training)
        return x


class IACSEncoder(nn.Module):
    def __init__(self, args, node_feat_dim, edge_feat_dim):
        super(IACSEncoder, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.query_size = args.task_size - args.num_shots
        self.num_shots = args.num_shots
        self.pool_type = args.pool_type
        self.film_type = args.film_type

        self.gnn = GNN(args, self.node_feat_dim, self.edge_feat_dim)
        if self.film_type in ["GATE", "gate"]:
            self.film = FiLM(self.gnn.num_out, use_gate=True)
        elif self.film_type in ["PLAIN", "plain"]:
            self.film = FiLM(self.gnn.num_out, use_gate=False)
        if self.pool_type in self.pool_type in ["Att", "att"]:
            self.attention = nn.MultiheadAttention(embed_dim=self.gnn.num_out, num_heads=1)
            self.LK = nn.Linear(self.gnn.num_out, self.gnn.num_out)
            self.LV = nn.Linear(self.gnn.num_out, self.gnn.num_out)
            self.LQ = nn.Linear(self.gnn.num_out, self.gnn.num_out)

    def forward(self, support_batch):
        x, edge_index, x_batch, edge_attr, bias = \
            support_batch.x, support_batch.edge_index, support_batch.batch, support_batch.edge_attr, support_batch.bias
        x_hid = self.gnn(x, edge_index, x_batch, edge_attr, bias)
        x_hid = x_hid.view(self.num_shots, -1, self.gnn.num_out)
        if self.pool_type in ["SUM", "sum"]:
            context = torch.sum(x_hid, dim=0, keepdim=False)
            context = torch.sum(x_hid, dim=0, keepdim=False)
        elif self.pool_type in ["AVG", "avg"]:
            context = torch.mean(x_hid, dim=0, keepdim=False)
            context = torch.mean(x_hid, dim=0, keepdim=False)
        elif self.pool_type in ["Att", "att"]: #todo
            Q = self.LQ(x_hid)
            V = self.LV(x_hid)
            K = self.LK(x_hid)
            attn_output, _ = self.attention(Q, K, V)
            context = torch.sum(attn_output, dim=0, keepdim=False)
        else:
            raise NotImplementedError("Unsupported Context Pooling type!")
        if self.film_type in ["GATE", "gate", "PLAIN", "plain"]:
            context = self.film(context)
        return context


class IACSIPDecoder(nn.Module):
    def __init__(self):
        super(IACSIPDecoder, self).__init__()
        self.linear =  MLP(in_ch=256, hid_ch=128, out_ch=128)

    def forward(self, context, query_batch):
        query, query_index, y, mask = query_batch.query, query_batch.query_index, query_batch.y, query_batch.mask
        attr_feats=query_batch.attr_feats

        q = context[query]
        q = scatter_mean(q, query_index, dim=0)

        #concate
        if attr_feats is not None:
            attr_feats=attr_feats.view(q.size(0),-1)
            q2=torch.cat((q,attr_feats),dim=1)
            q3=self.linear(q2)
            hid = torch.einsum("nc,kc->nk", [q3, context])
        else:
            hid = torch.einsum("nc,kc->nk", [q, context])
        hid = torch.flatten(hid)
        return hid, y, mask


class CSIACSComp(nn.Module):
    def __init__(self, args, node_feat_dim, edge_feat_dim, decoder_type: str= "MLP"):
        super(CSIACSComp, self).__init__()
        self.encoder = IACSEncoder(args, node_feat_dim, edge_feat_dim)
        self.decoder_type = decoder_type
        if decoder_type == "IP":
            self.decoder = IACSIPDecoder()
        else:
            raise NotImplementedError("Unsupported IACS Decoder type!")

    def forward(self, support_batch, query_batch, mode='spt'):
        context = self.encoder(support_batch) #137,128
        if mode=='spt':
            hid_spt, y_spt, mask_spt=self.decoder(context, support_batch)
            return hid_spt, y_spt, mask_spt
        elif mode=='qry':
            hid, y, mask=self.decoder(context, query_batch)
            return  hid, y, mask
        else:
            hid_spt, y_spt, mask_spt=self.decoder(context, support_batch)
            hid, y, mask=self.decoder(context, query_batch)
            return  hid_spt, y_spt, mask_spt, hid, y, mask


