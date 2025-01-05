# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# sentence = "Hello"
# inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
# print(inputs)
# # #------------------------------------------------------------------------------------------------#

import torch
from bert import BertModel

sanity_data = torch.load("./sanity_check.data")
# text_batch = ["hello world", "hello neural network for NLP"]

# tokenizer here
# above text_batch is tokenized using sentencepiece (refer to BERT paper for details)
sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                         [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1]])

# load our model
bert = BertModel.from_pretrained('bert-base-uncased')
outputs = bert(sent_ids, att_mask)
att_mask = att_mask.unsqueeze(-1)
outputs['last_hidden_state'] = outputs['last_hidden_state'] * att_mask
sanity_data['last_hidden_state'] = sanity_data['last_hidden_state'] * att_mask

for k in ['last_hidden_state', 'pooler_output']:
    assert torch.allclose(outputs[k], sanity_data[k], atol=1e-5, rtol=1e-3)
print("Your BERT implementation is correct!")

