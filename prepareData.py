import cv2
import csv
import os
import numpy as np

'''
writer = csv.writer(open('data_1_1043.csv', 'w'))
header = ['thermal(abs,sca)','class,mask']
writer.writerow(header)
'''




def load_images(folder_name, out_size=(64,48),single_channel=True):
	'''
	out_size = (X,Y)
	'''
	images = []
	for filename in os.listdir(folder_name):
		img = cv2.imread(os.path.join(folder_name,filename))
		if img is not None:
			img = cv2.resize(img,out_size)
			if single_channel:
				img = np.expand_dims(img[:,:,0],axis=-1)
			images.append(img)
	images = np.array(images).astype('float32')
	images = images/127.5 - 1
	return(images)




def load_images_from_folder0(folder_abs,folder_sca): # for input_data (thermal image) -> dim to (48,64,1) (now (48, 64, 3))
	images = []
	for filename in os.listdir(folder_abs):
		if img_abs is None or img_sca is None:
			raise ValueError('FUCK U')
		img_abs = cv2.imread(os.path.join(folder_abs,filename))
		img_sca = cv2.imread(os.path.join(folder_sca,filename))
		img_merge = np.concatanate([img_abs,img_sca],axis=-1)
		images.append(img_merge)
		'''
		if img_abs is not None and img_sca is not None:
			temp = []
			for idx_row,row in enumerate(img_abs):
				new_line = []
				for idx_pix,pixel in enumerate(row):
					new_pixel = []
					new_pixel.append(pixel[0])
					new_pixel.append(img_sca[idx_row][idx_pix][0])
					new_line.append(new_pixel)
				temp.append(new_line)
			images.append(temp)
		'''
	images = np.array(images).astype('float32')/255*2-1

	return images # (x imgs,48,64)

'''
def load_images_from_folder1(folder,input_class): # for input_mask (mask rcnn) -> resize! dim to [48,64,1] (now [480,640,3吧])
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		if img is not None:
			resized = cv2.resize(img, (64,48))
			temp = []
			for row in resized:
				new_line = []
				for pixel in row:
					new_line.append((pixel[0]/255)*2-1)
				temp.append(new_line)
			temp = np.array(temp)
			images.append([input_class,temp])
	image = np.array(images)
	return images

def create_black_mask(folder,input_class): # this folder(thermal) is for photo number counting
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		if img is not None:
			resized = cv2.resize(img, (64,48))
			temp = []
			for row in resized:
				new_line = []
				for pixel in row:
					new_line.append(-1)
				temp.append(new_line)
			temp = np.array(temp)
			images.append([input_class,temp])
	image = np.array(images)
	return images
'''

def load_images_from_folder1(folder,input_class): # for input_mask (mask rcnn) -> resize! dim to [48,64,1] (now [480,640,3吧])
	images = []
	temp_class = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		if img is not None:
			resized = cv2.resize(img, (64,48))
			temp = []
			for row in resized:
				new_line = []
				for pixel in row:
					new_line.append((pixel[0]/255)*2-1)
				temp.append(new_line)
			temp = np.array(temp)
			images.append(temp)
			temp_class.append(input_class)
	images = np.array(images)
	temp_class = np.array(np.array(images))
	return images, temp_class

def create_black_mask(folder,input_class): # this folder(thermal) is for photo number counting
	images = []
	temp_class = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		if img is not None:
			resized = cv2.resize(img, (64,48))
			temp = []
			for row in resized:
				new_line = []
				for pixel in row:
					new_line.append(-1)
				temp.append(new_line)
			temp = np.array(temp)
			images.append(temp)
			temp_class.append(input_class)
	images = np.array(images)
	temp_class = np.array(np.array(images))
	return images, temp_class

def get_data():
	
	print('hi')
	x_1043_abs = load_images('grideye_dataset/yes_human/temp/absrender_1632343831043')
	x_1043_sca = load_images('grideye_dataset/yes_human/temp/render_1632343831043')
	y_1043_class = np.ones((x_1043_abs.shape[0],1),dtype=np.float32)
	y_1043_mask = load_images('grideye_dataset/yes_human/rgb/results_1632343831043')
	#print(y_1043_mask.shape)
	#print(np.min(y_1043_mask))
	#print(np.max(y_1043_mask))
	print('hi')
	x_8483_abs = load_images('grideye_dataset/yes_human/temp/absrender_1632344048483')
	x_8483_sca = load_images('grideye_dataset/yes_human/temp/render_1632344048483')
	y_8483_class = np.ones((x_8483_abs.shape[0],1),dtype=np.float32)
	y_8483_mask = load_images('grideye_dataset/yes_human/rgb/results_1632344048483')
	print('hi')
	x_4898_abs = load_images('grideye_dataset/normal_human/temp/absrender_1632342424898')
	x_4898_sca = load_images('grideye_dataset/normal_human/temp/render_1632342424898')
	y_4898_class = np.zeros((x_4898_abs.shape[0],1),dtype=np.float32)
	y_4898_mask = load_images('grideye_dataset/normal_human/rgb/results_1632342424898')
	print('hi')
	x_1257_abs = load_images('grideye_dataset/normal_human/temp/absrender_1632342561257')
	x_1257_sca = load_images('grideye_dataset/normal_human/temp/render_1632342561257')
	y_1257_class = np.zeros((x_1257_abs.shape[0],1),dtype=np.float32)
	y_1257_mask = load_images('grideye_dataset/normal_human/rgb/results_1632342561257')
	print('hi')
	x_9067_abs = load_images('grideye_dataset/no_human/temp/absrender_1632342729067')
	x_9067_sca = load_images('grideye_dataset/no_human/temp/render_1632342729067')
	y_9067_class = np.zeros((x_9067_abs.shape[0],1),dtype=np.float32)
	y_9067_mask = np.zeros((x_9067_abs.shape[0],48,64,1),dtype=np.float32) - 1
	print('hi')
	x_6167_abs = load_images('grideye_dataset/no_human/temp/absrender_1632343066176')
	x_6167_sca = load_images('grideye_dataset/no_human/temp/render_1632343066176')
	y_6167_class = np.zeros((x_6167_abs.shape[0],1),dtype=np.float32)
	y_6167_mask = np.zeros((x_6167_abs.shape[0],48,64,1),dtype=np.float32) - 1

	x_abs = np.concatenate([x_1043_abs,x_8483_abs,x_4898_abs,x_1257_abs,x_9067_abs,x_6167_abs],axis=0)
	x_sca = np.concatenate([x_1043_sca,x_8483_sca,x_4898_sca,x_1257_sca,x_9067_sca,x_6167_sca],axis=0)
	y_class = np.concatenate([y_1043_class,y_8483_class,y_4898_class,y_1257_class,y_9067_class,y_6167_class],axis=0)
	y_mask = np.concatenate([y_1043_mask,y_8483_mask,y_4898_mask,y_1257_mask,y_9067_mask,y_6167_mask],axis=0)

	x_merged = np.concatenate([x_abs,x_sca],axis=-1)

	return x_merged, y_mask, y_class

	# 1 1043
	#input_abs = 'grideye_dataset/yes_human/temp/absrender_1632343831043' 
	#input_sca = 'grideye_dataset/yes_human/temp/render_1632343831043'
	#input_class = 1 #yes_human->1, no_human&normal_human->0
	#input_mask = 'grideye_dataset/yes_human/rgb/results_1632343831043'

	#thermal = load_images_from_folder0(input_abs,input_sca)
	#mask,class_data = load_images_from_folder1(input_mask,input_class)

	# 1 8483
	#input_abs = 'grideye_dataset/yes_human/temp/absrender_1632344048483' 
	#input_sca = 'grideye_dataset/yes_human/temp/render_1632344048483'
	#input_class = 1 #yes_human->1, no_human&normal_human->0
	#input_mask = 'grideye_dataset/yes_human/rgb/results_1632344048483'

	'''
	for i in load_images_from_folder0(input_abs,input_sca):
		thermal = np.append(thermal,i)
	t_mask,t_class_data = load_images_from_folder1(input_mask,input_class)
	for i in t_mask:
		mask = np.append(mask,i)
	for i in t_class_data:
		class_data = np.append(class_data,i)
	'''

	# 0 4898 normal human
	#input_abs = 'grideye_dataset/normal_human/temp/absrender_1632342424898' 
	#input_sca = 'grideye_dataset/normal_human/temp/render_1632342424898'
	#input_class = 0 #yes_human->1, no_human&normal_human->0
	#input_mask = 'grideye_dataset/normal_human/rgb/results_1632342424898'

	'''
	for i in load_images_from_folder0(input_abs,input_sca):
		thermal = np.append(thermal,i)
	'''
	'''
	for i in load_images_from_folder1(input_mask,input_class):
		mask.append(i)
	'''
	'''
	t_mask,t_class_data = load_images_from_folder1(input_mask,input_class)
	for i in t_mask:
		mask = np.append(mask,i)
	for i in t_class_data:
		class_data = np.append(class_data,i)
	'''

	# 0 1257 normal human
	#input_abs = 'grideye_dataset/normal_human/temp/absrender_1632342561257' 
	#input_sca = 'grideye_dataset/normal_human/temp/render_1632342561257'
	#input_class = 0 #yes_human->1, no_human&normal_human->0
	#input_mask = 'grideye_dataset/normal_human/rgb/results_1632342561257'

	'''
	for i in load_images_from_folder0(input_abs,input_sca):
		thermal = np.append(thermal,i)
	t_mask,t_class_data = load_images_from_folder1(input_mask,input_class)
	for i in t_mask:
		mask = np.append(mask,i)
	for i in t_class_data:
		class_data = np.append(class_data,i)
	'''

	# 0 9067 no human
	#input_abs = 'grideye_dataset/no_human/temp/absrender_1632342729067' 
	#input_sca = 'grideye_dataset/no_human/temp/render_1632342729067'
	#input_class = 0 #yes_human->1, no_human&normal_human->0
	'''
	for i in load_images_from_folder0(input_abs,input_sca):
		thermal = np.append(thermal,i)
	t_mask,t_class_data = load_images_from_folder1(input_mask,input_class)
	for i in t_mask:
		mask = np.append(mask,i)
	for i in t_class_data:
		class_data = np.append(class_data,i)
	'''

	# 0 6167 no human
	#input_abs = 'grideye_dataset/no_human/temp/absrender_1632343066176' 
	#input_sca = 'grideye_dataset/no_human/temp/render_1632343066176'
	#input_class = 0 #yes_human->1, no_human&normal_human->0

	'''
	for i in load_images_from_folder0(input_abs,input_sca):
		thermal = np.append(thermal,i)
	t_mask,t_class_data = load_images_from_folder1(input_mask,input_class)
	for i in t_mask:
		mask = np.append(mask,i)
	for i in t_class_data:
		class_data = np.append(class_data,i)

	return thermal,mask,class_data
	'''


'''
def check_file():
	# 0 4898 normal human mask has 61 shortage files in compare to rgb/thermal: 0-51,306-314
	input_abs = 'grideye_dataset/normal_human/temp/absrender_1632342424898' 
	input_sca = 'grideye_dataset/normal_human/temp/render_1632342424898'
	input_class = 0 #yes_human->1, no_human&normal_human->0
	input_mask = 'grideye_dataset/normal_human/rgb/results_1632342424898'
	input_rgb = 'grideye_dataset/normal_human/rgb/1632342424898'

	rgb_file = []
	mask_file = []
	for filename in os.listdir(input_mask):
		mask_file.append(filename)
	for filename in os.listdir(input_rgb):
		rgb_file.append(filename)

	for name in rgb_file:
		if name not in mask_file and name != '.DS_Store':
			temp = int(name.split('.')[0])
			if temp > 52:
				print(name)
'''

#get_data()
#check_file()
'''
for idx,i in enumerate(thermal):
	writer.writerow([i,mask[idx]])
'''