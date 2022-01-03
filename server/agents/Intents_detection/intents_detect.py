import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import json 
from transformers import AutoModel, BertTokenizerFast
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoModel, BertTokenizerFast
from flask import Flask, request, jsonify
from flask_cors import CORS

# specify GPU
USE_CUDA = torch.cuda.is_available()
device = torch.device("cpu")

# Converting the labels into encodings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Based on the histogram we are selecting the max len as 8
max_seq_len = 8

words=[]
classes = []
documents = []
ignore_words = ['?', '!']

data_file = open('server/agents/Intents_detection/intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #add documents in the corpus
        documents.append((pattern, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


df = pd.DataFrame(documents, columns = ["text", "label"])
df['label'] = le.fit_transform(df['label'])
# check class distribution
df['label'].value_counts(normalize = True)
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# Import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')


class BERT_Arch(nn.Module):
   def __init__(self, bert):      
       super(BERT_Arch, self).__init__()
       self.bert = bert 
      
       # dropout layer
       self.dropout = nn.Dropout(0.2)
      
       # relu activation function
       self.relu =  nn.ReLU()
       # dense layer
       self.fc1 = nn.Linear(768,512)
       self.fc2 = nn.Linear(512,256)
       self.fc3 = nn.Linear(256,9)
       #softmax activation function
       self.softmax = nn.LogSoftmax(dim=1)
       #define the forward pass
   def forward(self, sent_id, mask):
      #pass the inputs to the model  
      cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
      
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      
      x = self.fc2(x)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc3(x)
   
      # apply softmax activation
      x = self.softmax(x)
      return x

for param in bert.parameters():
      param.requires_grad = False
model = BERT_Arch(bert)
model.load_state_dict(torch.load("server/agents/Intents_detection/model.zip", device))
model = model.to(device)

def get_prediction(str, model):
    str = re.sub(r'[^a-zA-Z ]+', '', str)
    test_text = [str]
    model.eval()
    
    tokens_test_data = tokenizer(
    test_text,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])
    
    preds = None
    with torch.no_grad():
      preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis = 1)
    print("Intent Identified: ", le.inverse_transform(preds)[0])
    return le.inverse_transform(preds)[0]

def get_response(message, model): 
  intent = get_prediction(message, model)
  intents = json.loads(open('server/agents/Intents_detection/intents.json').read())
  for i in intents['intents']: 
    if i["tag"] == intent:
      result = random.choice(i["responses"])
      message={ "answer":[{"_type": "dialog",
                    "message": result}]}
      break
  #print(f"Response : {result}")
  return message


#get_response("i need my certificate work ", model)



app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
@app.route('/api/search/smart-agent/search/<term>', methods=['GET', 'POST'])
def test(term):
  chatbot_message = get_response(term, model)

  return jsonify(chatbot_message)
app.run(debug=True, port=9090)
