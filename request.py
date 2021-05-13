import numpy as np
from flask import Flask, request, jsonify, render_template
import csv

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
   
    with open('news_file.csv', mode='w') as csv_file:
        fieldnames = ['news', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'news': 'collh', 'label': 'Accooooooong'})
        writer.writerow({'news': 'Johnhhhhh', 'label': 'Accosssssssssg'})
        
    a=['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','fake','c']
         
    return  render_template('index.html',title1=a[0],label1=a[1]
                            ,title2=a[0],label2=a[1],title3=a[0],label3=a[1]
                            ,title4=a[0],label4=a[1],title5=a[0],label5=a[1]
                            ,title6=a[0],label6=a[1],title7=a[0],label7=a[1]
                            ,title8=a[0],label8=a[1],title9=a[0],label9=a[1]
                            ,title10=a[0],label10=a[1])


if __name__ == "__main__":
    app.run(debug=True)