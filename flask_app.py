import io

from flask import Flask, request, make_response, render_template
from scipy.io.wavfile import read, write
import inference

app = Flask(__name__)



@app.route('/', methods=['GET'])
def upload_form():
    return render_template('upload.html')


@app.route(f'/', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_name = file.filename
    file_bytes = file.read()
    memory_file = io.BytesIO(file_bytes)
    rate, song = read(memory_file)

    separated_stems = inference.separate_into_dict(song, rate=rate)
    zip_file = inference.write_dict_to_zip_in_memory(separated_stems)

    response = make_response(zip_file.getvalue())
    response.headers.set('Content-Type', 'application/zip')
    response.headers.set('Content-Disposition', 'attatchment', filename=f'{file_name} - Separated.zip')

    return response