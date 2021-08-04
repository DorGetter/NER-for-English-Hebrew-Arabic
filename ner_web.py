from flask import Flask ,request, render_template ,url_for,redirect
app = Flask(__name__)

data= "ner"

@app.route('/')
def start_form():
        return render_template("start.html")

@app.route('/', methods=['POST','GET'])
def second_form():
    if request.method=='POST':
        data=request.form['fname']
        ##here we call the mode and chang the data
        ##data=model(data)
        return render_template('start.html',data=data)
    else:
        
        return render_template('start.html',data=data)
