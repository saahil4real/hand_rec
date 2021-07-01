import cv2
import numpy as np
from flask import Flask, request
from flask_restx import Resource, Api, reqparse	
from werkzeug.datastructures import FileStorage
from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
from split_line import *
from format_change import *
from spell_check import *
from test import *
from combine import *



app = Flask('__name__')
api = Api(app)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


@api.route('/upload/')
@api.expect(upload_parser)
class Upload(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        uploaded_file.save('pic.jpg')
        file_path = 'pic.jpg'
        decoder_type = DecoderType.BestPath
        model = Model(list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True, dump=False)
        findings = infer(model, file_path)
        # img = cv2.imread(uploaded_file)
        # print(uploaded_file)
        # url = do_something_with_file(uploaded_file)
        return findings, 201

@api.route("/hello")
class HelloWorld(Resource):
	def get(self):
		return {'hello': 'world'}

if __name__ == '__main__':
	# This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
