from flask import Flask ,request, render_template 

app = Flask(__name__)



@app.route('/')
def start_form():
        return render_template("start.html")

@app.route('/', methods=['POST'])
def second_form():
    text=request.form['fname']
    print(text)
    return render_template('anser.html',dataToRender=text)

