from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle

app = Flask(__name__,template_folder='templ')
tf = pickle.load(open('tfvec.pkl','rb'))
model = pickle.load(open('spam.pkl','rb'))
# stem = pickle.load(open('stemmer.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        txt = request.form['text']
    # steming = stem(txt) 
    result = tf.transform([txt]).toarray()
    prediction = model.predict(result)

    output = (prediction)
    return render_template('index.html',prediction = 'prediction is {}'.format(output))
    
    # return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)