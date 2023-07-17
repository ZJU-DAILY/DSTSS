import subprocess
import random
from tqdm import tqdm

tr_len = 50  # 定长轨迹
num_queries = 20
input_path = 'geolife-data/geolife-r-speed-0.4'
output_path = 'geolife-data/geolife-r-speed-queries'


def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

# num_lines = file_len(input_path)
# num_trs = int(num_lines / tr_len)
num_trs = 400000

print(f'# of trajectories: {num_trs}')

indexes = random.sample(range(num_trs), num_queries)
max_index = max(indexes)

with open(input_path) as f, open(output_path, 'w') as g:
    for i, line in tqdm(enumerate(f), total=max_index*tr_len):
        tr_idx = int(i / 50)
        if tr_idx > max_index:
            break
        if tr_idx in indexes:
            g.write(line)
