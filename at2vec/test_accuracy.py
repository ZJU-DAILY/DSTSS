import torch
import math
from at2vec import PretrainModel, EncoderDecoder, get_mat
from gensim.models import Word2Vec
from functools import partial
import random
from heapq import heappush, heappop
from tqdm import tqdm, trange
from torch.multiprocessing import Pool
import pandas as pd
import os

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda')


# Configs
directory = 'tdrive-data/'
training_set_file = directory + 'tdrive-r-train-ps-50'
sp_pretrain_model_path = directory + 'sp_pretrain_model.pt'
ts_pretrain_model_path = directory + 'ts_pretrain_model.pt'
sm_pretrain_model_path = directory + 'semantic2vec.model'
at2vec_model_path = directory + 'at2vec_model.pt'
trajectory_file = directory + 'tdrive-r-speed-0.4-medium'
query_file = trajectory_file + '-lcss-result'

true_results = {}  # num_queries x maxk
with open(query_file) as f:
    ds_sizes = [int(ds_size) for ds_size in f.readline().strip().split()]
    for ds_size in ds_sizes:
        true_results[ds_size] = []
        
    num_queries = 100
    query_idxs = []
    for _ in range(num_queries):
        query_idxs.append(int(f.readline().strip()))
    for _ in range(num_queries):
        for ds_size in ds_sizes:
            topk = [int(pair.split(',')[0]) for pair in f.readline().strip().split()]
            true_results[ds_size].append(topk)
for key in true_results:
    true_results[key] = torch.tensor(true_results[key], device=device)
    
PROGRESS_INTERVAL = 10000

# ts_gap = 1
ts_gap = 10 * 60 * 1000
num_x_grids = 200
num_y_grids = 200
sp_len, ts_len, sm_len = 100, 100, 100
pt_len = sp_len + ts_len + sm_len
hidden_len = 256
sampled_tr_len, complete_tr_len = 40, 50


INT_MAX = 214748364700000000000
FLOAT_MAX = 1E8


# These params are needed for loading the model
num_sp_grids = None
num_ts_grids = None

# Recover range info from training set
ts_range = (INT_MAX, -INT_MAX)
x_range = (FLOAT_MAX, -FLOAT_MAX)
y_range = (FLOAT_MAX, -FLOAT_MAX)

with open(training_set_file) as f:
    for line in f:
        fields = line.strip().split('\t')
        ts = int(fields[1])
        x = float(fields[2])
        y = float(fields[3])
        ts_range = min(ts_range[0], ts), max(ts_range[1], ts)
        x_range = min(x_range[0], x), max(x_range[1], x)
        y_range = min(y_range[0], y), max(y_range[1], y)
        

def pair2spid(x_id: int, y_id: int, num_x_grids: int):
    return y_id * num_x_grids + x_id

def sp2id(x: float, y: float,
          min_x: float, min_y: float,
          max_x: float, max_y: float,
          x_gap: float, y_gap: float):
    """
    (x, y)坐标转换为空间网格令牌值。假设max_x和max_y不能取到。

    Returns:
        令牌值, (x轴编号, y轴编号)
    """
    x, y = max(min_x, x), max(min_y, y)
    x, y = min(max_x, x), min(max_y, y)
    num_x_grids = int(math.ceil((max_x - min_x) / x_gap))
    x_grid, y_grid = (int(math.floor((x - min_x) / x_gap)),
                      int(math.floor((y - min_y) / y_gap)))
    return pair2spid(x_grid, y_grid, num_x_grids)

print('ts_range', ts_range)
num_ts_grids = (ts_range[1] - ts_range[0]) // ts_gap + 1
x_gap = (x_range[1] - x_range[0]) / num_x_grids
y_gap = (y_range[1] - y_range[0]) / num_y_grids
num_sp_grids = sp2id(x_range[1], y_range[1],
                     x_range[0], y_range[0],
                     x_range[1], y_range[1],
                     x_gap, y_gap)

# load the model
sp_model = PretrainModel(num_sp_grids, sp_len, torch.device('cpu'))
sp_model.load_state_dict(torch.load(sp_pretrain_model_path)['model'])
ts_model = PretrainModel(num_ts_grids, ts_len, torch.device('cpu'))
ts_model.load_state_dict(torch.load(ts_pretrain_model_path)['model'])
sm_model = Word2Vec.load(sm_pretrain_model_path)

model = EncoderDecoder(sampled_tr_len, complete_tr_len, pt_len, hidden_len,
                       num_sp_grids, num_ts_grids, len(sm_model.wv), device)
state = torch.load(at2vec_model_path)
model.load_state_dict(state['model'])

print('model loaded')

# 读取轨迹文件
class BareDataset(torch.utils.data.Dataset):
    def __init__(self, tr_path="", ntrs=None):
        def read_data(path: str, nrows: int):
            if not path:
                return None, None
            data = pd.read_csv(path, sep="\t", nrows=nrows,
                               usecols=[0, 1, 2, 3, 4], header=None)
            idx = dict()  # tid -> [start_index, len]
            current_tid = -1
            total = data.shape[0]
            print('total lines: ' + str(total))
            # 假设每条轨迹在文件中是连续的
            for line_no, (i, point) in tqdm(enumerate(data.iterrows()), total=data.shape[0], disable=True):
                if line_no % (50 * PROGRESS_INTERVAL) == 0:
                    print(f'{line_no} / {data.shape[0]} lines processed')
                tid = point.iloc[0]
                if current_tid != tid:
                    idx[tid] = [i, 0]
                    current_tid = tid
                idx[tid][1] += 1
            return data, idx

        nrows = None if ntrs is None else ntrs * complete_tr_len
        # 读取原始轨迹数据
        self.data, self.data_idx = read_data(tr_path, nrows=nrows)

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError
            
        def get_from(data, data_idx, index):
            if data is None:
                return None
            start_idx, length = data_idx[index]
            return data.iloc[start_idx:start_idx+length]

        return get_from(self.data, self.data_idx, index)


# load data
def getTestTensorDataset(bare_dataset, raw2tr):
    raws = []
    vectors = [None] * len(bare_dataset)
    for i in tqdm(range(len(bare_dataset))):
        raw = bare_dataset[i]
        raws.append(raw)
    with Pool() as p:
        total = len(bare_dataset)
        for start in tqdm(range(0, len(bare_dataset), 250), disable=True):
            if (start % PROGRESS_INTERVAL == 0):
                print(f"{start} / {total} trajectories processed")
            end = min(start + 250, len(bare_dataset))
            vectors[start:end] = p.map(raw2tr, raws[start:end])
    return torch.stack(vectors, dim=0)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, bare_dataset, raw2tr):
        self.bare_dataset = bare_dataset
        self.vectors = []
        
        raws = []
        for i in tqdm(range(len(self.bare_dataset)), disable=True):
            #if i % 100000 == 0:
            #    print('TestDataset: processing raw data #' + str(i))
            raw = self.bare_dataset[i]
            raws.append(raw)
        with Pool() as p:
            self.vectors = p.map(raw2tr, raws)

    def __len__(self):
        return len(self.bare_dataset)

    def __getitem__(self, index):
        """
        Returns:
            (index, tr)
        """
        tr = self.vectors[index]
        return index, tr


def ts2id(ts: int, min_ts: int, max_ts: int, ts_gap: int):
    ts = min(max_ts, ts)
    ts = max(min_ts, ts)
    return int((ts - min_ts) // ts_gap)

# Raw -> tr
def get_mat(tr, sp_model, ts_model, sm_model):
    get_spid = partial(sp2id, min_x=x_range[0], max_x=x_range[1], min_y=y_range[0], max_y=y_range[1],
                       x_gap=x_gap, y_gap=y_gap)
    get_tsid = partial(
        ts2id, min_ts=ts_range[0],  max_ts=ts_range[1], ts_gap=ts_gap)
    ts_col, all_cols, sm_col = (tr.iloc[:, 1],
                                tr.iloc[:, 2:4],
                                tr.iloc[:, 4])
    sp_vec = torch.stack([sp_model.embed(get_spid(al.iloc[0], al.iloc[1]))
                          for (_, al) in all_cols.iterrows()], dim=0)
    ts_vec = torch.stack([ts_model.embed(get_tsid(ts))
                         for (_, ts) in ts_col.iteritems()], dim=0)
    # semantics are more complicated
    vec_set = []
    for _, sm in sm_col.iteritems():
        # For each trajectory point
        # keyword list of this point
        kws = sm.replace(' ', '-').split(',')
        # 所有关键词向量取平均并归一化，作为该点语义向量
        avg_vec = torch.from_numpy(sm_model.wv.get_mean_vector(
            kws, pre_normalize=True, post_normalize=True))
        vec_set.append(avg_vec)
    sm_vec = torch.stack(vec_set, dim=0)
    # returns: (tr_len, sp_len)
    return torch.cat((sp_vec, ts_vec, sm_vec), dim=1)


raw2tr = partial(get_mat, sp_model=sp_model,
                 ts_model=ts_model, sm_model=sm_model)

bare_dataset = BareDataset(trajectory_file)
print('bare dataset ready')
try:
    torch.save(bare_dataset, directory + 'bare_dataset.pt')
    print('bare dataset saved')
except:
    print('failed to save bare dataset.')
    try:
        os.remove(directory + 'bare_dataset.pt')
    except:
        pass

    
# 正式测试
min_ds_size = min(ds_sizes)
max_ds_size = max(ds_sizes)

def get_true_results(idx):
    min_idx = idx // 50 * 50
    max_idx = min_idx + 50
    return range(min_idx, max_idx)


def get_accuracy(items, targets):
    # intersections size
    unique, count = torch.unique(torch.cat((items, targets)), return_counts=True)
    return unique[count > 1].numel()
    
# ds_size -> k -> query
precision = dict()  
recall = dict()

ks = [5, 10, 20, 50]
for ds_size in ds_sizes:
    precision[ds_size] = {}
    recall[ds_size] = {}
    for k in ks:
        precision[ds_size][k] = []
        recall[ds_size][k] = []

bare_queries = []

for query_idx in query_idxs:
    bare_queries.append(bare_dataset[query_idx])
with Pool() as p:
    queries = p.map(raw2tr, bare_queries)
queries = model.get_rep_vector(torch.stack(queries, dim=0).to(device))  # (num_queries, vec_size)
        
dists_array = []

print('calculating distance...')
p = Pool()
total = len(bare_dataset)
for start in tqdm(range(0, len(bare_dataset), 500), disable=True):
    if start % PROGRESS_INTERVAL == 0:
        print(f'{start} / {total} trajectories calculated')
    end = min(start + 500, len(bare_dataset))
    raws = []
    for i in range(start, end):
        raws.append(bare_dataset[i])
    # trs: (batch_size, tr_len)
    trs = p.map(raw2tr, raws)
    vecs = model.get_rep_vector(torch.stack(trs).to(device))  # (batch_size, vec_size)
    # (num_queries, batch_size)
    # print(queries.shape, vecs.shape)
    dist = torch.cdist(queries.unsqueeze(0), vecs.unsqueeze(0)).squeeze()
    dists_array.append(dist)
dists = torch.cat(dists_array, dim=1)  # (num_queries, tr_count)
print('completed.')
p.close()

print('generating statistics:')

for ds_size in ds_sizes:
    maxk = max(ks)
    topk = torch.topk(dists[:, :ds_size], dim=1, k=maxk+1, largest=False, sorted=True)[1]  # indices (num_queries,)
    for query_idx, true_result, result in zip(query_idxs, true_results[ds_size], topk):
        print(true_result.shape)
        for k in ks:
            precision[ds_size][k].append(get_accuracy(result[1:k+1], true_result[:k]) / k)
            recall[ds_size][k].append(get_accuracy(result[1:k+1], true_result) / k)

print('precision:', precision, sep="\n")
# print('recall:', recall, sep="\n")

print('-----')
print('AVERAGE')

print('ds_size k precision@k recall@k')
for ds_size in ds_sizes:
    for k in ks:
        print(ds_size, k, sum(precision[ds_size][k]) / len(precision[ds_size][k]),
             sum(recall[ds_size][k]) / len(recall[ds_size][k]))
