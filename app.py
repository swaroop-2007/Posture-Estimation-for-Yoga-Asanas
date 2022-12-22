from glob import glob
#import pwd
from flask import Flask, render_template, Response
import cv2
import os
import time
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from numpi import yoga
from model import siamies
import torch
from torch import nn, tensor

glob_vals=[]

UPLOAD_FOLDER = '.\\inference\\input_directory'
app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

@app.route('/')
def index():
    return render_template('index.html')

def gen():

    cap = cv2.VideoCapture('output.mp4')

    while(cap.isOpened()):
        ret, img = cap.read()
        if ret ==True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
            time.sleep(0.1)

        else:
            break

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_upload')
def video_upload():
	return render_template('video_upload.html')
	
@app.route('/video_upload_process',methods=['GET','POST'])
def video_upload_process():	
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		# file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		# print('upload_video filename: ' + filename)
		# flash('Video successfully uploaded and displayed below')
		
		# os.system(f'move {filename}.mp4 C:\\Users\\swapn\\Documents\\BE_project\\BE_project\\video_pose\\VideoPose3D\\inference\\input_directory')
		# os.chdir('./inference')
		
		# os.system('python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --output-dir output_directory --image-ext mp4 input_directory')
		# os.chdir('..')
		# os.system('dir')
		# os.chdir('./data')

		# os.system('python prepare_data_2d_custom.py -i C:/Users/swapn/Documents/BE_project/BE_project/video_pose/VideoPose3D/inference/output_directory -o myvideos')
		# os.chdir('..')
		# os.system(f'''python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject {filename} --viz-action custom --viz-camera 0 --viz-video ./inference/input_directory/{filename} --viz-output export/output.mp4 --viz-size 6''')
		# os.system(f'''python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject {filename} --viz-action custom --viz-camera 0 --viz-video ./inference/input_directory/{filename} --viz-export export/{filename} --viz-size 6''')
		os.chdir('./export')
		loaded_arr = np.load(f'{filename}.npy')
		
		yg_obj = yoga()
		inp = yg_obj.transform(loaded_arr,4)

		inp = inp[None,None,:,:]
		model_obj = siamies()
		model_obj.load_state_dict(torch.load('seimies_state_dict.pth'))
		model_obj = model_obj.eval()

		inp_embed = model_obj(inp)
		print(inp_embed.shape)

		inp_embed = inp_embed.view(1,-1)

		pwd = nn.PairwiseDistance(p=2)
		dist_arr = []
		min_dist = None
		min_arr = None
		min_dist2 = None

		for file in os.listdir(r'./'):
			if file.endswith('.pt'):
				temp_tensor = torch.load(file)
				temp_tensor = temp_tensor.view(1,-1)
				
				temp_dist = pwd(inp_embed,temp_tensor)
				if min_dist == None:
					min_dist = temp_dist
				else:
					if min_dist > temp_dist:
						min_dist = temp_dist
					else:
						if min_dist2 == None:
							min_dist2 = temp_dist
							min_arr = temp_tensor
						else:
							if min_dist2 > temp_dist:
								min_dist2 = temp_dist
								min_arr = temp_tensor
				
				

		cos = nn.CosineSimilarity(dim=1 , eps=1e-6)
		output = cos(inp_embed,min_arr)
		print(output)

		global glob_vals

		print(min_dist2)
		print(min_arr)
		print(inp_embed)

		inp_embed = torch.round(inp_embed[0], decimals=2)
		min_arr = torch.round(min_arr[0], decimals=2)

		glob_vals.append(min_dist2)
		glob_vals.append(output)
		glob_vals.append(inp_embed)
		glob_vals.append(min_arr)

		return redirect(url_for('video_play'))

@app.route('/video_play')
def video_play():
	global glob_vals
	return render_template('video_play.html', pwd=glob_vals[0], cos=glob_vals[1], usr_embed=glob_vals[2], db_embed=glob_vals[3])

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/contact')
def contact():
	return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
