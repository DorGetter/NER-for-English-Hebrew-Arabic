from flask import Flask, request, json
from run_aleph import look_and_see 
from flask_cors import CORS, cross_origin

app = Flask("aleph_server")

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

Flag = False

@app.route('/new_code', methods=['POST'])
def check():
    data = json.loads(request.data.decode("utf-8"))
    look_and_see(data["name"])
    return "sucess"
    
app.run(host='0.0.0.0', port=106)
  