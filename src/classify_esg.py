from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import pickle

results = []

with open('../dataset/docs.pkl', 'rb') as fp:
    docs, docs_info = pickle.load(fp)
    
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

dataset = ListDataset(docs)
    
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer, truncation=True, max_length=256, device=0)

for out in tqdm(nlp(dataset, batch_size=128), total=len(dataset)):
    results.append(out)
    
with open('../dataset/esg_classification.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)