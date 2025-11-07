import os
import pandas as pd
from datasets import Dataset
from utils import read_json

class MediQDataset():
    def __init__(self, data_dir, specialty, split_idx):
        self.data_dir = data_dir
        self.specialty = specialty
        self.split_idx = split_idx
    
    def load(self):
        data_lst = []
        for split in ['train', 'test', 'cal']:
            path = os.path.join(self.data_dir, 'splits', self.specialty, f'split{self.split_idx}_{split}.csv')
            df = pd.read_csv(path, index_col=0)
            ds = Dataset.from_pandas(df)            

            # add column of queries
            converted_queries = []
            for sample_i in range(len(ds)):
                path = f'./results/mediq/convert/{self.specialty}/llama3.1-8b/split{self.split_idx}/{split}_sample{sample_i}.json'
                converted_queries.append(read_json(path)['questions'])
            ds = ds.add_column('queries', converted_queries)
            data_lst.append(ds)
        self.data_lst = data_lst
        return data_lst
    
if __name__ == '__main__':
    
    mediq_dataset = MediQDataset('./data/mediQ/', 'internal_medicine', 0)
    train_dataset, _, _ = mediq_dataset.load()
    import pdb; pdb.set_trace()