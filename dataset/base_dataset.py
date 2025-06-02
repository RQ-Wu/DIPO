import os, sys
import json
import numpy as np
# import collections.abc
# sys.modules['collections'].Mapping = collections.abc.Mapping

import networkx as nx
from torch.utils.data import Dataset
from my_utils.refs import cat_ref, sem_ref, joint_ref, data_mode_ref
from collections import deque

def build_graph(tree, K=32):
    '''
    Function to build graph from the node list.
    
    Args:
        nodes: list of nodes
        K: the maximum number of nodes in the graph
    Returns:
        adj: adjacency matrix, records the 1-ring relationship (parent+children) between nodes
        edge_list: list of edges, for visualization
    '''
    adj = np.zeros((K, K), dtype=np.float32)
    parents = []
    tree_list = []
    for node in tree:
        tree_list.append(
            {
                'id': node['id'],
                'parent_id': node['parent'],
            }
        )
        # 1-ring relationship
        if node['parent'] != -1:
            adj[node['id'], node['parent']] = 1
            parents.append(node['parent'])
        else:
            adj[node['id'], node['id']] = 1
            parents.append(-1)
        for child_id in node['children']:
            adj[node['id'], child_id] = 1 
    return {
        'adj': adj,
        'parents': np.array(parents, dtype=np.int8),
        'tree_list': tree_list
    }

from collections import defaultdict
from functools import cmp_to_key

def bfs_tree_simple(tree_list):
    order = [0] * len(tree_list)
    queue = []
    current_node_idx = 0
    for node_idx, node in enumerate(tree_list):
        if node['parent_id'] == -1:
            queue.append(node['id'])
            order[node_idx] = current_node_idx
            current_node_idx += 1
            break
    while len(queue) > 0:
        current_node = queue.pop(0)
        for node_idx, node in enumerate(tree_list):
            if node['parent_id'] == current_node:
                queue.append(node['id'])
                order[node_idx] = current_node_idx
                current_node_idx += 1

    return order

def bfs_tree(tree_list, aabb_list, epsilon=1e-3):
    # 初始化遍历顺序列表
    order = [0] * len(tree_list)
    current_order = 0
  
    # 构建父节点到子节点的索引映射
    parent_map = defaultdict(list)
    for idx, node in enumerate(tree_list):
        parent_map[node['parent_id']].append(idx)
  
    # 查找根节点
    root_indices = [idx for idx, node in enumerate(tree_list) if node['parent_id'] == -1]
    if not root_indices:
        return order
  
    # 初始化队列（存储节点索引）
    queue = [root_indices[0]]
    order[root_indices[0]] = current_order
    current_order += 1

    # 比较函数：按中心坐标排序
    def compare_centers(a, b):
        # 获取两个节点的中心坐标
        center_a = [(aabb_list[a][i] + aabb_list[a][i+3])/2 for i in range(3)]
        center_b = [(aabb_list[b][i] + aabb_list[b][i+3])/2 for i in range(3)]
      
        # 逐级比较坐标（考虑epsilon阈值）
        for coord in range(3):
            delta = abs(center_a[coord] - center_b[coord])
            if delta > epsilon:
                return -1 if center_a[coord] < center_b[coord] else 1
        return 0  # 所有坐标差均小于阈值时保持原顺序

    # BFS遍历
    while queue:
        current_idx = queue.pop(0)
        current_id = tree_list[current_idx]['id']
      
        # 获取子节点索引并排序
        children = parent_map.get(current_id, [])
        sorted_children = sorted(children, key=cmp_to_key(compare_centers))
      
        # 处理子节点
        for child_idx in sorted_children:
            order[child_idx] = current_order
            current_order += 1
            queue.append(child_idx)

    return order

class BaseDataset(Dataset):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
    
    def _filter_models(self, models_ids):
        '''
        Filter out models that has more than K nodes.
        '''
        json_data_root = self.hparams.json_root
        filtered = []
        for i, model_id in enumerate(models_ids):
            if i % 100 == 0:
                print(f'Checking model {i}/{len(models_ids)}')
            path = os.path.join(json_data_root, model_id, self.json_name)
            with open(path, 'r') as f:
                json_file = json.load(f)
                if len(json_file['diffuse_tree']) <= self.hparams.K:
                    filtered.append(model_id)
        return filtered
    
    def get_acd_mapping(self):
        self.category_mapping = {
            'armoire': 'StorageFurniture',
            'bookcase': 'StorageFurniture',
            'chest_of_drawers': 'StorageFurniture',
            'desk': 'Table',
            'dishwasher': 'Dishwasher',
            'hanging_cabinet': 'StorageFurniture',
            'kitchen_cabinet': 'StorageFurniture',
            'microwave': 'Microwave',
            'nightstand': 'StorageFurniture',
            'oven': 'Oven',
            'refrigerator': 'Refrigerator',
            'sink_cabinet': 'StorageFurniture',
            'tv_stand': 'StorageFurniture',
            'washer': 'WashingMachine',
            'table': 'Table',
            'cabinet': 'StorageFurniture',
            'hanging_cabinet': 'StorageFurniture',
        }

    def _random_permute(self, graph, nodes):
        '''
        Function to randomly permute the nodes and update the graph and node attribute info.

        Args:
            graph: a dictionary containing the adjacency matrix, edge list, and root node
            nodes: a list of nodes
        Returns:
            graph_permuted: a dictionary containing the updated adjacency matrix, edge list, and root node
            nodes_permuted: a list of permuted nodes
        '''
        N = len(nodes)
        order = np.random.permutation(N)
        graph_permuted = self._reorder_nodes(graph, order)
        exchange = [0] * len(order)
        for i in range(len(order)):
            exchange[order[i]] = i
        nodes_permuted = nodes[exchange, :]
        return graph_permuted, nodes_permuted
    
    def _permute_by_order(self, graph, nodes, order):
        '''
        Function to permute the nodes and update the graph and node attribute info by order.

        Args:
            graph: a dictionary containing the adjacency matrix, edge list, and root node
            nodes: a list of nodes
            order: a list of indices for reordering
        Returns:
            graph_permuted: a dictionary containing the updated adjacency matrix, edge list, and root node
            nodes_permuted: a list of permuted nodes
        '''
        graph_permuted = self._reorder_nodes(graph, order)
        if nodes is None:
            return graph_permuted, None
        else:
            exchange = [0] * len(order)
            for i in range(len(order)):
                exchange[order[i]] = i
            nodes_permuted = nodes[exchange, :]
            return graph_permuted, nodes_permuted
    
    def _prepare_node_data(self, node):
        # semantic label
        label = np.array([sem_ref['fwd'][node['name']]], dtype=np.float32) / 5. - 0.8 # (1,), range from -0.8 to 0.8
        # joint type
        joint_type = np.array([joint_ref['fwd'][node['joint']['type']] / 5.], dtype=np.float32) - 0.5 # (1,), range from -0.8 to 0.8
        # aabb
        aabb_center = np.array(node['aabb']['center'], dtype=np.float32)  # (3,), range from -1 to 1
        aabb_size = np.array(node['aabb']['size'], dtype=np.float32) # (3,), range from -1 to 1
        aabb_max = aabb_center + aabb_size / 2
        aabb_min = aabb_center - aabb_size / 2
        # joint axis and range
        if node['joint']['type'] == 'fixed':
            axis_dir = np.zeros((3,), dtype=np.float32)
            axis_ori = aabb_center
            joint_range = np.zeros((2,), dtype=np.float32)
        else:
            if node['joint']['type'] == 'revolute' or node['joint']['type'] == 'continuous':
                joint_range = np.array([node['joint']['range'][1]], dtype=np.float32) / 360. 
                joint_range = np.concatenate([joint_range, np.zeros((1,), dtype=np.float32)], axis=0) # (2,) 
            elif node['joint']['type'] == 'prismatic' or node['joint']['type'] == 'screw':
                joint_range = np.array([node['joint']['range'][1]], dtype=np.float32) 
                joint_range = np.concatenate([np.zeros((1,), dtype=np.float32), joint_range], axis=0) # (2,) 
            axis_dir = np.array(node['joint']['axis']['direction'], dtype=np.float32) * 0.7 # (3,), range from -0.7 to 0.7
            # make sure the axis is pointing to the positive direction
            if np.sum(axis_dir > 0) < np.sum(-axis_dir > 0): 
                axis_dir = -axis_dir 
                joint_range = -joint_range
            axis_ori = np.array(node['joint']['axis']['origin'], dtype=np.float32) # (3,), range from -1 to 1
            if (node['joint']['type'] == 'prismatic' or node['joint']['type'] == 'screw') and node['name'] != 'door':
                axis_ori = aabb_center
        # prepare node data by given mod name
        # aabb = np.concatenate([aabb_max, aabb_min], axis=0)
        # axis = np.concatenate([axis_dir, axis_ori], axis=0)
        # node_data_all = [aabb, joint_type.repeat(6), axis, joint_range.repeat(3), label.repeat(6)]
        # node_data_list = [node_data_all[data_mode_ref[mod_name]] for mod_name in self.hparams.data_mode]
        # node_data = np.concatenate(node_data_list, axis=0)
        node_label = np.ones(6, dtype=np.float32)

        node_data = np.concatenate([aabb_max, aabb_min, joint_type.repeat(6), axis_dir, axis_ori, joint_range.repeat(3), label.repeat(6), node_label], axis=0)
        if self.hparams.mode_num == 5:
            node_data = np.concatenate([aabb_max, aabb_min, joint_type.repeat(6), axis_dir, axis_ori, joint_range.repeat(3), label.repeat(6)], axis=0)
        return node_data


    def _reorder_nodes(self, graph, order):
        '''
        Function to reorder nodes in the graph and 
        update the adjacency matrix, edge list, and root node.

        Args:
            graph: a dictionary containing the adjacency matrix, edge list, and root node
            order: a list of indices for reordering
        Returns:
            new_graph: a dictionary containing the updated adjacency matrix, edge list, and root node
        '''
        N = len(order)
        mapping = {i: order[i] for i in range(N)}
        mapping.update({i: i for i in range(N, self.hparams.K)})
        G = nx.from_numpy_array(graph['adj'], create_using=nx.Graph)
        G_ = nx.relabel_nodes(G, mapping)
        new_adj = nx.adjacency_matrix(G_, G.nodes).todense()

        exchange = [0] * len(order)
        for i in range(len(order)):
            exchange[order[i]] = i
        return {
            'adj': new_adj.astype(np.float32),
            'parents': graph['parents'][exchange]
        }


    def _prepare_input_GT(self, file, model_id):
        '''
        Function to parse input item from a json file for the CAGE training.
        '''
        tree = file['diffuse_tree']
        K = self.hparams.K # max number of nodes
        cond = {} # conditional information and axillary data
        cond['parents'] = np.zeros(K, dtype=np.int8)

        # prepare node data
        nodes = []
        for node in tree:
            node_data = self._prepare_node_data(node) # (36,)
            nodes.append(node_data) 
        nodes = np.array(nodes, dtype=np.float32)
        n_nodes = len(nodes)

        # prepare graph
        graph = build_graph(tree, self.hparams.K)
        if self.mode == 'train': # perturb the node order for training
            graph, nodes = self._random_permute(graph, nodes)

        # pad the nodes to K with empty nodes
        if n_nodes < K:
            empty_node = np.zeros((nodes[0].shape[0],))
            data = np.concatenate([nodes, [empty_node] * (K - n_nodes)], axis=0, dtype=np.float32) # (K, 36)
        else:
            data = nodes
        mode_num = data.shape[1] // 6
        data = data.reshape(K*mode_num, 6) # (K * n_attr, 6)

        # attr mask (for Local Attention)
        attr_mask = np.eye(K, K, dtype=bool)
        attr_mask = attr_mask.repeat(mode_num, axis=0).repeat(mode_num, axis=1)
        cond['attr_mask'] = attr_mask

        # key padding mask (for Global Attention)
        pad_mask = np.zeros((K*mode_num, K*mode_num), dtype=bool)
        pad_mask[:, :n_nodes*mode_num] = 1
        cond['key_pad_mask'] = pad_mask

        # adj mask (for Graph Relation Attention)
        adj_mask = graph['adj'][:].astype(bool)
        adj_mask = adj_mask.repeat(mode_num, axis=0).repeat(mode_num, axis=1)
        adj_mask[n_nodes*mode_num:, :] = 1
        cond['adj_mask'] = adj_mask

        # object category
        if self.map_cat:  # for ACD dataset
            category = file['meta']['obj_cat']
            category = self.category_mapping[category]
            cond['cat'] = cat_ref[category]
        else:
            cond['cat'] = cat_ref.get(file['meta']['obj_cat'], None)
            if cond['cat'] is None:
                cond['cat'] = self.category_mapping.get(file['meta']['obj_cat'], None)
                if cond['cat'] is None:
                    cond['cat'] = 2
                else:
                    cond['cat'] = cat_ref.get(cond['cat'], None)
            # cond['cat'] = cat_ref[file['meta']['obj_cat']]
        if cond['cat'] is None:
            cond['cat'] = 2
        # axillary info
        cond['name'] = model_id
        cond['adj'] = graph['adj']
        cond['parents'][:n_nodes] = graph['parents']
        cond['n_nodes'] = n_nodes
        cond['obj_cat'] = file['meta']['obj_cat']
        
        return data, cond

    def _prepare_input(self, model_id, pred_file, gt_file=None):
        '''
        Function to parse input item from pred_file, and parse GT from gt_file if available.
        '''
        K = self.hparams.K # max number of nodes
        cond = {} # conditional information and axillary data
        # prepare node data
        n_nodes = len(pred_file['diffuse_tree'])
        # prepare graph
        pred_graph = build_graph(pred_file['diffuse_tree'], K)
        # dummy GT data
        data = np.zeros((K*5, 6), dtype=np.float32)
        
        # attr mask (for Local Attention)
        attr_mask = np.eye(K, K, dtype=bool)
        attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['attr_mask'] = attr_mask

        # key padding mask (for Global Attention)
        pad_mask = np.zeros((K*5, K*5), dtype=bool)
        pad_mask[:, :n_nodes*5] = 1
        cond['key_pad_mask'] = pad_mask

        # adj mask (for Graph Relation Attention)
        adj_mask = pred_graph['adj'][:].astype(bool)
        adj_mask = adj_mask.repeat(5, axis=0).repeat(5, axis=1)
        adj_mask[n_nodes*5:, :] = 1
        cond['adj_mask'] = adj_mask

        # placeholder category, won't be used if category is given (below)
        cond['cat'] = cat_ref['StorageFurniture']
        cond['obj_cat'] = 'StorageFurniture'
        # if object category is given as input
        if not self.hparams.get('test_label_free', False):
            assert 'meta' in pred_file, 'meta not found in the json file.'
            assert 'obj_cat' in pred_file['meta'], 'obj_cat not found in the metadata of the json file.'
            category = pred_file['meta']['obj_cat']
            if self.map_cat:  # for ACD dataset
                category = self.category_mapping[category]
            cond['cat'] = cat_ref[category]
            cond['obj_cat'] = category

        # axillary info
        cond['name'] = model_id
        cond['adj'] = pred_graph['adj']
        cond['parents'] = np.zeros(K, dtype=np.int8)
        cond['parents'][:n_nodes] = pred_graph['parents']
        cond['n_nodes'] = n_nodes

        return data, cond

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

