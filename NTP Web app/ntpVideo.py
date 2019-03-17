import os
from flask import Flask,url_for,redirect,render_template, request,send_from_directory
from werkzeug import secure_filename

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from object_detection1 import ObjectDetection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
from PIL import Image
# Import utilites
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
from tensorflow import DType
import math
import reverse_geocoder as rg 
#import pprint
import datetime
import MySQLdb
import time
import imageio

#from flask_restful import Resource, Api
#api = Api(app)
UPLOAD_FOLDER='C:/Users/HP/Desktop/models-master/research/object_detection/PWP webapp/static/img/'
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

MODEL_FILENAME = 'frozen_inference_graph.pb'
LABELS_FILENAME = 'labels.txt'

current_location=""
class TFlowObjectDetection(ObjectDetection):
	
	def __init__(self, graph_def, labels):
		super(TFlowObjectDetection, self).__init__(labels)
		self.graph = tf.Graph()
		layers=[n.name for n in tf.get_default_graph().as_graph_def().node]
		print(layers)
		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')

			
	def predict(self, preprocessed_image):
		inputs = np.array(preprocessed_image, dtype=np.float)[:,:,(2,1,0)] # RGB -> BGR

		with tf.Session(graph=self.graph) as sess:
			output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
			outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis,...]})
			return outputs[0]


def convert_to_opencv(image):
	# RGB -> BGR conversion is performed as well.
	r,g,b = np.array(image).T
	opencv_image = np.array([b,g,r]).transpose()
	return opencv_image

def crop_center(img,cropx,cropy):
	h, w = img.shape[:2]
	startx = w//2-(cropx//2)
	starty = h//2-(cropy//2)
	return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
	h, w = image.shape[:2]
	if (h < 1600 and w < 1600):
		return image

	new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
	return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
	h, w = image.shape[:2]
	return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
	exif_orientation_tag = 0x0112
	if hasattr(image, '_getexif'):
		exif = image._getexif()
		if (exif != None and exif_orientation_tag in exif):
			orientation = exif.get(exif_orientation_tag, 1)
			# orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
			orientation -= 1
			if orientation >= 4:
				image = image.transpose(Image.TRANSPOSE)
			if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
				image = image.transpose(Image.FLIP_TOP_BOTTOM)
			if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
				image = image.transpose(Image.FLIP_LEFT_RIGHT)
	return image

def reverseGeocode(coordinates): 
	result = rg.search(coordinates) 
	  
	# result is a list containing ordered dictionary. 
	return result

def connection():
	conn = MySQLdb.connect(host="localhost",
		user="aditi",
		passwd="ads296",
		db="pwp5")

	c=conn.cursor()
	return c, conn

def pred_video():

		graph_def_obj = tf.GraphDef()
		with tf.gfile.FastGFile(MODEL_FILENAME, 'rb') as f:
			graph_def_obj.ParseFromString(f.read())

		# Load labels
		with open(LABELS_FILENAME, 'r') as f:
			labels = [l.strip() for l in f.readlines()]

		od_model = TFlowObjectDetection(graph_def_obj, labels)
		#image = Image.open(UPLOAD_FOLDER+filename)
		#imageWhole = image
		#imagenp=np.array(image,dtype=np.uint8)
		input_video='vid3'
		video_reader=imageio.get_reader('%s.mp4'%input_video)
		video_writer=imageio.get_writer('%s_annotated.mp4'% input_video,fps=10)
		t0= datetime.datetime.now()
		n_frames=0
		f=0
		font = ImageFont.truetype("arial.ttf", 80)
		for frame in video_reader:

			startX=0.0
			starty=0.0
			endX=0.0
			endY=0.0
			c=0
			d=0
			w=0
			h=0
			
			#img_np_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			#image_np=img_np_bgr
			heightP,widthP= frame.shape[:2]#image_np.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
			#heightP = image_np.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
			print("WIDTH:",widthP,"\nHEIGHT:",heightP)
			#draw = ImageDraw.Draw(image_np)
			n_frames+=1
			pil_im = Image.fromarray(frame)
			draw = ImageDraw.Draw(pil_im)
			predictions1 = od_model.predict_image(pil_im)
			print(predictions1)
			print(type(predictions1))
			pred_out_list=[]
			pred_prob=0.0
			j=0
			for p in predictions1:
				for d1 in predictions1:
					if(d1['probability']>0.35):
						print(d1['boundingBox'].items())
						for n1,n2 in d1['boundingBox'].items():
							#image=imageWhole
							if(n1=='left'):
								startX=float(n2)*widthP
								l=float(n2)
							elif(n1=='top'):
								startY=float(n2)*heightP
								t=float(n2)
							elif(n1=='width'):
								endX=startX+float(n2)*widthP
								w=float(n2)
							elif(n1=='height'):
								endY=startY+float(n2)*heightP
								h=float(n2)
					print("left: ",startX,"\ntop: ",startY,"\nwidth: ",endX,"\nheight: ",endY)
					draw.rectangle(((startX, startY), (endX, endY)), fill=None, width=10, outline="red")
					draw.text((2, (100*j)+100),d1['tagName'],(255,0,0),font=font)
					j=j+1
				open_cv_image2 = np.array(pil_im) 
				img_np_bgr2 = cv2.cvtColor(open_cv_image2, cv2.COLOR_RGB2BGR)
				#video_writer=imageio.get_writer('%s_output.mp4'%input_video,fps=10)
				video_writer.append_data(img_np_bgr2)
				# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (widthP,heightP),1)
				# out.write(img_np_bgr2)
			#cv2.imshow('video',img_np_bgr2)
		#font = ImageFont.truetype("arial.ttf", 300)
		'''for p in predictions1:
			if(p['probability']>0.3):
				pred_prob=str(round(p['probability']*100,2))
				pred_out_dict[p['tagName']]=pred_prob
		print(pred_out_dict)'''

if __name__=='__main__':
	pred_video()