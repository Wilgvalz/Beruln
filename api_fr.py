from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import torch 
from fastapi import FastAPI, Request, Response

app = FastAPI()
class model_input(BaseModel):
    text_list : str

import pickle
import io
import torch
import transformers

g = 'vuln_job_1.pkl'

class CPU_Unpickler(pickle.Unpickler):
    g = 'vuln_job_1.pkl'
    def find_class(self, module, name):
        g = 'vuln_job_1.pkl'
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

with open(g,'rb') as f:
    vuln_model = CPU_Unpickler(f).load()
 

@app.post('/')

async def vuln_pred(input_parameters:str):
    
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    import pandas as pd 
    from keras.utils import pad_sequences
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer, BertConfig

    input_dictionary = input_parameters
    sd = {1:input_dictionary}
    df = pd.DataFrame(list(sd.items()), columns=['labels', 'func'])
    
    sentences = df['func']


    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df['labels']

    from transformers import BertTokenizer, BertModel
    tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenized_texts2 = [tokenizer2.tokenize(sent) for sent in sen2]
    tokenized_texts = [tokenizer2.tokenize(sent) for sent in sentences]

    MAX_LEN = 128

    input_ids = [tokenizer2.convert_tokens_to_ids(x) for x in tokenized_texts]

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    batch_size = 1

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    import numpy as np

    def softmax(logits):
        e = np.exp(logits)
        return e / np.sum(e)
     
    import torch
    import numpy as np

    vuln_model.eval()

    raw_predictions, predicted_classes, true_labels = [], [], []

    for batch in prediction_dataloader:
        device = 'cpu'
        batch = tuple(t.to(device) for t in batch)
  
        b_input_ids, b_input_mask, b_labels = batch
  
        with torch.no_grad():
    
            outputs = vuln_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs['logits'].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        b_input_ids = b_input_ids.to('cpu').numpy()
        batch_sentences = [tokenizer2.decode(input_ids, skip_special_tokens=True) for input_ids in b_input_ids]

        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
  
        batch_predictions = np.argmax(probabilities, axis=1)

  
        for i, sentence in enumerate(batch_sentences):
            dsa = batch_predictions[i]
        if dsa == 1:
            dsa = 'vuln'
        if dsa == 0:
            dsa = 'unvuln'
        
    import torch
    from transformers import AutoTokenizer
    
    import numpy as np
    model_checkpoint = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space = True)
    id2label = {1:'vuln', 0:'unvuln'}
     
    predictions = dsa 
    if predictions == 'vuln': #.item()
        return 'Double-check your code, there may be a vulnerability in it!'
    else:
        return "I don't see any vulnerabilites in your code"                   