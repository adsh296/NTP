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
from flask import Flask
from flask_mail import Mail, Message


#from flask_restful import Resource, Api
#api = Api(app)
UPLOAD_FOLDER='C:/Users/HP/Desktop/models-master/research/object_detection/PWP webapp/static/img/'
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
mail=Mail(app)

MODEL_FILENAME = 'frozen_inference_graph.pb'
LABELS_FILENAME = 'labels.txt'

current_location=""
class Brand_detection(ObjectDetection):

	def __init__(self, graph_def, labels):
		super(Brand_detection, self).__init__(labels)
		self.graph = tf.Graph()
		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')
			
	def predict(self, preprocessed_image):
		inputs = np.array(preprocessed_image, dtype=np.float)[:,:,(2,1,0)] # RGB -> BGR

		with tf.Session(graph=self.graph) as sess:
			output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
			outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis,...]})
			return outputs[0]


def convert_to_opencv(image):
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


app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'pwp5deepblue@gmail.com'
app.config['MAIL_PASSWORD'] = 'pwp@2329'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True


@app.route("/")
def login():
    return render_template('userform.html')

@app.route('/home',methods=['POST'])
def savepost():
    a=request.form['user']
    return render_template('home.html',username=a)

@app.route('/home/results',methods=['POST'])
def uploadImage():
	pic=request.files['pic']
	
	j=0
	
	if request.method == 'POST':
		f = request.files['pic']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

		graph_def_obj = tf.GraphDef()
		with tf.gfile.FastGFile(MODEL_FILENAME, 'rb') as f:
			graph_def_obj.ParseFromString(f.read())

		# Load labels
		with open(LABELS_FILENAME, 'r') as f:
			labels = [l.strip() for l in f.readlines()]

		od_model = Brand_detection(graph_def_obj, labels)
		image = Image.open(UPLOAD_FOLDER+filename)
		imageWhole = image
		imagenp=np.array(image,dtype=np.uint8)
		predictions1 = od_model.predict_image(image)
		print(predictions1)
		print(type(predictions1))
		pred_out_list=[]
		pred_prob=0.0
		draw = ImageDraw.Draw(imageWhole)
		#font = ImageFont.truetype("arial.ttf", 300)
		'''for p in predictions1:
			if(p['probability']>0.3):
				pred_prob=str(round(p['probability']*100,2))
				pred_out_dict[p['tagName']]=pred_prob
		print(pred_out_dict)'''
		for p in predictions1:
			if(p['probability']>0.35):
				pred_prob=str(round(p['probability']*100,2))
				pred_out_list.append((p['tagName'],pred_prob))
		print(pred_out_list)

		orgInfo=image.info

		startX=0.0
		startY=0.0
		endX=0.0
		endY=0.0
		widthP,heightP=image.size
		c=0
		d=0
		w=0
		h=0
		print("WIDTH:",widthP,"\nHEIGHT:",heightP)
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

		# draw.text((x, y),"Sample Text",(r,g,b))
		'''for (i,p) in enumerate(pred_out_list):
			label = "{}%".format(p)
			draw.text((5, (800*i)+200),label,(255,255,255),font=font)'''
		#imageWhole_np=np.array(imageWhole,dtype=uint8)
		'''cv2.cvtColor(imageWhole_np,cv2.COLOR_BGR2RGB)
		for (i,p) in enumerate(pred_out_list):
				label = "{}%".format(p)
				cv2.putText(imageWhole, label, (10, (800*i)+200), 
					cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 10)'''

		try:
			c, conn = connection()
			print("okay")
		except Exception as e:
			print(str(e))

			current_datetime=datetime.datetime.now()
			print(current_datetime)
			formatted_date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
			print("fd: ",formatted_date)
			
			#label = "{0}".format(str(labels[highest_probability_index]))
			#draw.text((5, (600*j)+200),label,(255,0,0),font=font)
			#j=j+1
			
			

			'''
			try:
				c, conn = connection()
				print("okay")
				global current_location
				#sql="CREATE TABLE IF NOT EXISTS brands_track (dateAndtime datetime, brand varchar(25), location varchar(100))"
				loc=current_location #returnLoc()
				print("hey!"+loc)
				sql_insert="INSERT INTO brands_track(dateAndtime,brand,location) VALUES('%s','%s','%s')"%(formatted_date,labels[highest_probability_index],loc)
				c.execute(sql_insert)
				conn.commit()

			except Exception as e:
				print(str(e))
			'''

			'''for (i,p) in pred_out_dict.items():
				if(labels[highest_probability_index]==i):
					label = "{0}_{1}%".format(str(labels[highest_probability_index]),str(p))
					draw.text((5, (600*j)+200),label,(255,255,255),font=font)
					j=j+1
				else:
					label = "{0}".format(str(labels[highest_probability_index]))
					draw.text((5, (600*j)+200),label,(255,255,255),font=font)
					j=j+1'''
			#out_img = abs(np.fft.rfft2(out_img,axes=(0,1)))
			#out_img=np.uint8(out_img)
			#im = Image.fromarray(cm.gist_earth(out_img))
			#im.save("C:/Users/HP/Desktop/models-master/research/object_detection/PWP webapp/static/img/output_{}".format(filename),"JPEG",exif=orgInfo['exif'])
		
			#f.save(os.path.join(app.config['UPLOAD_FOLDER'],image))
		time.sleep(5)
		for tup in pred_out_list:
			try:
				global current_location
				#sql="CREATE TABLE IF NOT EXISTS brands_track (dateAndtime datetime, brand varchar(25), location varchar(100))"
				loc=current_location #returnLoc()
				print("hey!"+loc)
				sql_insert="INSERT INTO brands_track(dateAndtime,brand,location) VALUES('%s','%s','%s')"%(formatted_date,tup[0],loc)
				c.execute(sql_insert)
				conn.commit()

			except Exception as e:
				print(str(e))

		print(bool(orgInfo))
		if bool(orgInfo)!=False:
			imageWhole.save("C:/Users/HP/Desktop/models-master/research/object_detection/PWP webapp/static/img/output_{}".format(filename),"JPEG")

		else:
			imageWhole.save("C:/Users/HP/Desktop/models-master/research/object_detection/PWP webapp/static/img/output_{}".format(filename),"JPEG")
		msg1='Predictions' #"Here is the uploaded image"
		drawGraph()
		time.sleep(5)
	return render_template('result.html',pic='output_{}'.format(filename),msg1=msg1,brands=pred_out_list)

@app.route("/")
def send_mail():

	msg = Message('Hello', sender = 'pwp5deepblue@gmail.com', recipients = ['phoenix32h@gmail.com'])
	msg.body = "Hello "
	with app.open_resource("C:/Users/HP/Desktop/models-master/research/object_detection/PWP webapp/static/img/brand_graph.png") as fp:
		msg.attach("brand_graph.png", "image/png", fp.read())
	mail.send(msg)
	return render_template('mail_ack.html')

@app.route('/home/results/location',methods=['POST'])
def storeLocation():
	loc=""
	dict_loc={}
	global current_location
	if request.method == 'POST':
		user_location = request.get_json()
		print(user_location)
		lat=user_location['Latitude']
		lng=user_location['Longitude']
		coordinates=(lat,lng)
		loct=reverseGeocode(coordinates)
		print("loct: ",loct)
		for l in loct:
			print(l)
			dict_loc=dict(l)
			loc=dict_loc['name']
			current_location=loc
			print(current_location)
	return ''

def returnLoc():
	return current_location

def drawGraph():
	distinct_locations=[]
	distinct_brands=[]
	brands_records={}
	brands_records_percent={}
	loc_brands={}
	loc_brands_percent={}
	means_brnd_dict={}
	try:
		c, conn = connection()
		sql_dist="SELECT DISTINCT location FROM brands_track"
		c.execute(sql_dist)
		distinct_locations=list(c.fetchall())
		print(distinct_locations)
		sql_brands="SELECT DISTINCT brand FROM brands_track"
		c.execute(sql_brands)
		distinct_brands=list(c.fetchall())
		print(distinct_brands)
		for t1 in distinct_locations:
			total_packs=0
			for b1 in distinct_brands:
				brand_count=0
				sql_retrieve="SELECT COUNT(brand) FROM brands_track WHERE brand='%s' AND location='%s'"%(str(b1[0]),str(t1[0]))
				c.execute(sql_retrieve)
				brand_count=c.fetchone()
				#print(brand_count[0])
				brands_records[b1[0]]=brand_count[0]
				total_packs=total_packs+brand_count[0]
				
			print(brands_records)
			loc_brands[t1[0]]=brands_records
			print(total_packs)
			for br,cnt in brands_records.items():
				brands_records_percent[br]=(cnt/total_packs)*100
			print(brands_records_percent)
			d=0
			for brnd,pr in brands_records_percent.items():
				if(brnd==distinct_brands[d][0]):
					if(brnd in means_brnd_dict):
						means_brnd_dict[brnd]=means_brnd_dict[brnd]+(pr,)
						d=d+1
					else:
						means_brnd_dict[brnd]=(pr,)
						d=d+1
					
			print(means_brnd_dict)
			loc_brands_percent[t1[0]]=brands_records_percent
			print(loc_brands)
			print(loc_brands_percent)

		# data to plot
		n_groups = len(loc_brands_percent.items())
		print(n_groups)
		fig, ax = plt.subplots()
		color_list=['blue','red','green','black','cyan']
		gap = .8 / len(means_brnd_dict)
		i=0
		for yb,row in means_brnd_dict.items():
			X = np.arange(len(row))
			rects=plt.bar(X + i * gap, row, width = gap, color = color_list[i % len(color_list)], label=yb)
			i=i+1

		loc_list=[]
		for l in distinct_locations:
			loc_list.append(l[0])

		print(loc_list)

		plt.xlabel('Location')
		plt.ylabel('Brands Percentage')
		plt.title('Percentage of Brands in Different Locations')
		plt.xticks(np.arange(n_groups) + 0.3, tuple(loc_list))
		plt.legend()


		# create plot
		'''fig, ax = plt.subplots()
		index = np.arange(n_groups)
		bar_width = 0.35
		opacity = 0.8
		color_list=['blue','red','green','black','cyan']
		g=0	 
		
		
		rects1 = plt.bar(index, means_brnd_dict['CocaCola'], bar_width,
		alpha=opacity,
		color=color_list[0],
		label='CocaCola')
		 
		rects2 = plt.bar(index + bar_width, means_brnd_dict['Kurkure'], bar_width,
		alpha=opacity,
		color=color_list[1],
		label='Kurkure')

		rects3 = plt.bar(index + bar_width, means_brnd_dict['Lays'], bar_width,
		alpha=opacity,
		color=color_list[2],
		label='Lays')

		rects4 = plt.bar(index + bar_width, means_brnd_dict['Balaji'], bar_width,
		alpha=opacity,
		color=color_list[3],
		label='Balaji')

		rects2 = plt.bar(index + bar_width, means_brnd_dict['Sprite'], bar_width,
		alpha=opacity,
		color=color_list[4],
		label='Sprite')

		#rects2 = plt.bar(index + bar_width, means_guido, bar_width,
		#alpha=opacity,
		#color='g',
		#label='Guido')

		loc_list=[]
		for l in distinct_locations:
			loc_list.append(l[0])

		print(loc_list)
			 
		plt.xlabel('Location')
		plt.ylabel('Brands Percentage')
		plt.title('Percentage of Brands in Different Locations')
		plt.xticks(index + bar_width, tuple(loc_list))
		plt.legend()
		 
		plt.tight_layout()'''
		brndPic="C:/Users/HP/Desktop/models-master/research/object_detection/PWP webapp/static/img/brand_graph2.png"
		if os.path.exists(brndPic):
			os.remove(brndPic)
			fig.savefig("C:/Users/HP/Desktop/models-master/research/object_detection/PWP webapp/static/img/brand_graph2.png")
		else:
			fig.savefig("C:/Users/HP/Desktop/models-master/research/object_detection/PWP webapp/static/img/brand_graph2.png")

	except Exception as e:
		print(str(e))
	#return "/static/img/brand.png"

if __name__ == '__main__':
	#py_compile.compile('qswebapp.py')
	app.run(debug=True)