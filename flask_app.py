from flask import Flask
from flask import request
from flask import render_template
import test_backend_flask


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        test_backend_flask.test(l='a mers')
        return ('<p> I am a chill guy!</p>'
                '<form method="post" enctype="multipart/from-data">'
                '<label for="file">Choose file to upload</label>'
                '<input type="file" id="file" name="file">'
                '<button>Submit</button>'
                '</form>')
    if request.method == 'POST':
        img = request.files['file']
        return render_template('show_image_test.html', image=img)
