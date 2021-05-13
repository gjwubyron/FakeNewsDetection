import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification


import requests
from bs4 import BeautifulSoup
import re

import pickle
import csv

def GetRelatedNews(key):
    url='https://www.politifact.com/search/?q='+key
    
    page=requests.get(url)

    soup=BeautifulSoup(page.content, 'html.parser')

    links= soup.find_all('a',attrs={'href':re.compile("^/factchecks/20")},limit=10)
    
    titles=[]

    for link in links:
       
        URL='https://www.politifact.com'+str(link.get('href'))
        
        page=requests.get(URL)

        soup=BeautifulSoup(page.content, 'html.parser')
        title = soup.find('div',class_="m-statement__quote")
        
        titles.append(title.text)
        
    return titles

app = Flask(__name__)

with open('BERT_Model.pkl', 'rb') as file:  
    Pickled_Model = pickle.load(file)
    
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=2
                                                      )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.load_state_dict(torch.load(Pickled_Model, map_location=torch.device('cpu')))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    key=list(request.form.values())
    
    if (len(key[0].split())<=5):
        titles=GetRelatedNews(key[0])
    
        labels=[]
        with open(key[0]+'_file.csv', mode='w') as csv_file:
                fieldnames = ['news', 'label']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                
                writer.writeheader()
       
        for title in titles:
        
            inputs = tokenizer(title, return_tensors="pt")
            inputs.to(device)
            #Batch size 1
            outputs = model(**inputs)
            logits=outputs[0]
            pred = logits.detach().cpu().numpy()
            pred=np.argmax(pred, axis=1)
            if pred==0:
                label='Fake'
            else:
                label='True'
            labels.append(label)
            with open(key[0]+'_file.csv', mode='a+') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'news': title, 'label': label})
        while (len(titles)<10):
            titles.append(None)
        while (len(labels)<10):
            labels.append(None)
                
        return render_template('index.html',title1=titles[0],label1=labels[0]
                        ,title2=titles[1],label2=labels[1],title3=titles[2],label3=labels[2]
                        ,title4=titles[3],label4=labels[3],title5=titles[4],label5=labels[4]
                        ,title6=titles[5],label6=labels[5],title7=titles[6],label7=labels[6]
                        ,title8=titles[7],label8=labels[7],title9=titles[8],label9=labels[8]
                        ,title10=titles[9],label10=labels[9])
    else:
         inputs = tokenizer(key[0], return_tensors="pt")
         inputs.to(device)
         #Batch size 
         outputs = model(**inputs)
         logits=outputs[0]
         pred = logits.detach().cpu().numpy()
         pred=np.argmax(pred, axis=1)
         if pred==0:
             label='fake'
         else:
            label='true'
            
         return render_template('index.html',title1=key[0],label1=label)
    
   

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    
    key=list(list(request.form.values()))
    
    titles=GetRelatedNews(key[0].split())
    article=[]
    for title in titles:
   
        inputs = tokenizer(title, return_tensors="pt")
        inputs.to(device)
        #Batch size 1
        outputs = model(**inputs)
        logits=outputs[0]
        pred = logits.detach().cpu().numpy()
        pred=np.argmax(pred, axis=1)
        if pred==0:
            label='fake'
        else:
            label='true'
        statement=title+'is'+label
        article.append(statement)

    output=article
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    


