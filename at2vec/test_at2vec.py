import import_ipynb
from at2vec import ctx, sp2id
import torch
import heapq
from random import Random
import math


# 训练集参数
ctx.min_x = 116.15073
ctx.max_x = 116.60244
ctx.min_y = 39.749557
ctx.max_y = 40.101036
ctx.min_ts = 1201930254000
ctx.max_ts = 1202462476000
ctx.num_ts_grids = (ctx.max_ts - ctx.min_ts) // ctx.ts_gap + 1
ctx.x_gap, ctx.y_gap = ((ctx.max_x - ctx.min_x) / ctx.num_x_grids,
                        (ctx.max_y - ctx.min_y) / ctx.num_y_grids)
ctx.num_sp_grids = sp2id(ctx.max_x, ctx.max_y,
                         ctx.min_x, ctx.min_y,
                         ctx.max_x, ctx.max_y,
                         ctx.x_gap, ctx.y_gap)

ctx.test_tr_path = 'data/tdrive-sample-0.01-50x-ps-50'


class Evaluator:
    """
    用于检测单个数据集、单个噪音，并查询前k个结果。
    """
    def __init__(self, model, noise, k):
        self.model = model
        self.noise = noise
        self.k = k
        self.chosen_tr = None
        self.queue = []  # list[distance, index] 大根堆
    
    def set_chosen_tr(self, tr):
        """
        设置基准轨迹的向量表示。
        
        Args:
            tr: 基准轨迹的向量表示
        """
        self.chosen_tr_vec = model.get_rep_vector(tr)
    
    def evaluate(self, index: int, raw, tr):
        """
        先对轨迹加噪音，然后保存并返回离基准轨迹的距离。
        
        Args:
            index: 轨迹编号
            raw: pd.DataFrame 轨迹的原始表示
            tr: torch.Tensor  轨迹的向量表示
        Returns:
            添加噪音后的轨迹离基准轨迹的距离
        """
        d = torch.dist(model.get_rep_vector(noise.apply(raw, tr)), self.chosen_tr_vec)
        heapq.heappush(self.queue, (-d, index))
        if len(self.queue) > self.k:
            heapq.heappop(self.queue)
        return d
    
    def get_top_k_indexes(self):
        return sorted(self.queue)
    
class Noise:      
    def apply(self, raw, tr):
        """
        对轨迹加噪音。
        
        Args:
            raw: pd.DataFrame 轨迹的原始表示
            tr: torch.Tensor  轨迹的向量表示 (tr_len, sp_len)
        Returns:
            torch.Tensor 轨迹加噪音后的向量表示
        """
        raise NotImplemented
        
class NoNoise(Noise):
    def apply(self, raw, tr):
        return tr
    
    def __str__(self):
        return "no-noise"
    
    def __repr__(self):
        return "NoNoise()"
    

class PointSample(Noise):
    def __init__(self, p: float, rand: Random):
        """
        Args:
            p: 删掉一个轨迹点的概率
            rand: Random对象
        """
        assert 0 < p <= 1
        self.probability = p
        self.rand = rand
        
    def apply(self, raw, tr):
        tr_len = tr.shape[0]
        indexes = self.rand.sample(range(tr_len), k=math.round(tr_len * p))
        return tr[indexes]
    
    def __str__(self):
        return f'sample-{self.probability}'
    
    def __repr__(self):
        return f'PointSample(p={self.probability})'
    

class PointShift(Noise):
    def __init__(self, width: int, rand: Random):
        self.width = width
        self.rand = rand
        
    def apply(self, raw, tr):
        tr_len = tr.shape[0]
        width = self.width
        indexes = [i for i in range(tr_len)]
        for i in range(width, tr_len - width):
            # i下标的点应该移动到to
            to = i + rand.randrange(2 * width + 1) - width
            if to < i:
                temp = indexes[to]
                for j in range(to, i):
                    indexes[j] = indexes[j + 1]
                indexes[i] = temp
            elif to > i:
                temp = indexes[to]
                for j in reversed(i + 1, to + 1):
                    indexes[j] = indexes[j - 1]
                indexes[i] = temp
        return tr[indexes]
        
    def __str__(self):
        return (f'shift-{self.width}')
    
    def __repr__(self):
        return f'PointShift(width={self.width})'
    
class SemanticsSubstitution(Noise):
    def __init__(self,
                 p: float,
                 keyword_set,
                 raw_to_tr,
                 rand: Random):
        """
        Args:
            p: 替换语义信息的概率
            keyword_set: 可用关键词集合
            raw_to_tr: 原始表示转化为向量表示的函数
            rand: Random对象
        """
        self.probability = p
        self.keyword_set = keyword_set
        self.rand = rand
        self.raw_to_tr = raw_to_tr
    
    def apply(self, raw, tr):
        tr_len = raw.shape[0]
        raw_copy = raw.copy()
        for i in tr_len:
            keywords = raw_copy.iloc[i, 4].split(',')
            for j in range(len(keywords)):
                if rand.random() < probability:
                    keywords[i] = self.rand.sample(self.keyword_set)
            raw_copy.iloc[i, 4] = ','.join(keywords)
        return self.raw_to_tr(raw_copy)
    
    def __str__(self):
        return (f'substitute-{self.probability}')
    
    def __repr__(self):
        return (f'SemanticsSubstitution(p={self.probability})')
    
    
# 准备模型与数据
sp_model = PretrainModel(ctx.num_sp_grids, ctx.sp_len, torch.device('cpu'))
sp_model.load_state_dict(torch.load(ctx.sp_pretrain_model_path)['model'])
ts_model = PretrainModel(ctx.num_ts_grids, ctx.ts_len, torch.device('cpu'))
ts_model.load_state_dict(torch.load(ctx.ts_pretrain_model_path)['model'])
sm_model = Word2Vec.load(ctx.sm_pretrain_model_path)

bare_dataset = BareDataset(None, ctx.test_tr_path, update_ctx=False, ctx=ctx)
dataset = TrajectoryDataset(bare_dataset, sp_model, ts_model, sm_model, ctx)

model = EncoderDecoder(ctx.sampled_tr_len, ctx.complete_tr_len, ctx.pt_len, ctx.hidden_len,
                       ctx.num_sp_grids, ctx.num_ts_grids, len(sm_model.wv), torch.device('cpu'))
state = torch.load(ctx.at2vec_model_path)
model.load_state_dict(state['model'])

# 测试
noise = NoNoise()
evaluator = Evaluator(model=model, noise=noise)
total = len(dataset)
for i in range(1000):
    _, raw = bare_dataset[i]
    index, _, tr = dataset[i]
    evaluate.evaluate(index, raw, tr)
