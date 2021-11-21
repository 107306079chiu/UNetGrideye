



#FILTERINNG
#NO SCALING
#GRIDEYE CROP OUTPUT



import serial
import numpy as np
import time
import glob
import cv2
import collections
import threading
import os



mypath = './grideye_dataset/yes_human/temp'
write = True

end = False

temp_values = []

iron_palette = ["#00000a","#000014","#00001e","#000025","#00002a","#00002e","#000032","#000036","#00003a","#00003e","#000042","#000046","#00004a","#00004f","#000052","#010055","#010057","#020059","#02005c","#03005e","#040061","#040063","#050065","#060067","#070069","#08006b","#09006e","#0a0070","#0b0073","#0c0074","#0d0075","#0d0076","#0e0077","#100078","#120079","#13007b","#15007c","#17007d","#19007e","#1b0080","#1c0081","#1e0083","#200084","#220085","#240086","#260087","#280089","#2a0089","#2c008a","#2e008b","#30008c","#32008d","#34008e","#36008e","#38008f","#390090","#3b0091","#3c0092","#3e0093","#3f0093","#410094","#420095","#440095","#450096","#470096","#490096","#4a0096","#4c0097","#4e0097","#4f0097","#510097","#520098","#540098","#560098","#580099","#5a0099","#5c0099","#5d009a","#5f009a","#61009b","#63009b","#64009b","#66009b","#68009b","#6a009b","#6c009c","#6d009c","#6f009c","#70009c","#71009d","#73009d","#75009d","#77009d","#78009d","#7a009d","#7c009d","#7e009d","#7f009d","#81009d","#83009d","#84009d","#86009d","#87009d","#89009d","#8a009d","#8b009d","#8d009d","#8f009c","#91009c","#93009c","#95009c","#96009b","#98009b","#99009b","#9b009b","#9c009b","#9d009b","#9f009b","#a0009b","#a2009b","#a3009b","#a4009b","#a6009a","#a7009a","#a8009a","#a90099","#aa0099","#ab0099","#ad0099","#ae0198","#af0198","#b00198","#b00198","#b10197","#b20197","#b30196","#b40296","#b50295","#b60295","#b70395","#b80395","#b90495","#ba0495","#ba0494","#bb0593","#bc0593","#bd0593","#be0692","#bf0692","#bf0692","#c00791","#c00791","#c10890","#c10990","#c20a8f","#c30a8e","#c30b8e","#c40c8d","#c50c8c","#c60d8b","#c60e8a","#c70f89","#c81088","#c91187","#ca1286","#ca1385","#cb1385","#cb1484","#cc1582","#cd1681","#ce1780","#ce187e","#cf187c","#cf197b","#d01a79","#d11b78","#d11c76","#d21c75","#d21d74","#d31e72","#d32071","#d4216f","#d4226e","#d5236b","#d52469","#d62567","#d72665","#d82764","#d82862","#d92a60","#da2b5e","#da2c5c","#db2e5a","#db2f57","#dc2f54","#dd3051","#dd314e","#de324a","#de3347","#df3444","#df3541","#df363d","#e0373a","#e03837","#e03933","#e13a30","#e23b2d","#e23c2a","#e33d26","#e33e23","#e43f20","#e4411d","#e4421c","#e5431b","#e54419","#e54518","#e64616","#e74715","#e74814","#e74913","#e84a12","#e84c10","#e84c0f","#e94d0e","#e94d0d","#ea4e0c","#ea4f0c","#eb500b","#eb510a","#eb520a","#eb5309","#ec5409","#ec5608","#ec5708","#ec5808","#ed5907","#ed5a07","#ed5b06","#ee5c06","#ee5c05","#ee5d05","#ee5e05","#ef5f04","#ef6004","#ef6104","#ef6204","#f06303","#f06403","#f06503","#f16603","#f16603","#f16703","#f16803","#f16902","#f16a02","#f16b02","#f16b02","#f26c01","#f26d01","#f26e01","#f36f01","#f37001","#f37101","#f37201","#f47300","#f47400","#f47500","#f47600","#f47700","#f47800","#f47a00","#f57b00","#f57c00","#f57e00","#f57f00","#f68000","#f68100","#f68200","#f78300","#f78400","#f78500","#f78600","#f88700","#f88800","#f88800","#f88900","#f88a00","#f88b00","#f88c00","#f98d00","#f98d00","#f98e00","#f98f00","#f99000","#f99100","#f99200","#f99300","#fa9400","#fa9500","#fa9600","#fb9800","#fb9900","#fb9a00","#fb9c00","#fc9d00","#fc9f00","#fca000","#fca100","#fda200","#fda300","#fda400","#fda600","#fda700","#fda800","#fdaa00","#fdab00","#fdac00","#fdad00","#fdae00","#feaf00","#feb000","#feb100","#feb200","#feb300","#feb400","#feb500","#feb600","#feb800","#feb900","#feb900","#feba00","#febb00","#febc00","#febd00","#febe00","#fec000","#fec100","#fec200","#fec300","#fec400","#fec500","#fec600","#fec700","#fec800","#fec901","#feca01","#feca01","#fecb01","#fecc02","#fecd02","#fece03","#fecf04","#fecf04","#fed005","#fed106","#fed308","#fed409","#fed50a","#fed60a","#fed70b","#fed80c","#fed90d","#ffda0e","#ffda0e","#ffdb10","#ffdc12","#ffdc14","#ffdd16","#ffde19","#ffde1b","#ffdf1e","#ffe020","#ffe122","#ffe224","#ffe226","#ffe328","#ffe42b","#ffe42e","#ffe531","#ffe635","#ffe638","#ffe73c","#ffe83f","#ffe943","#ffea46","#ffeb49","#ffeb4d","#ffec50","#ffed54","#ffee57","#ffee5b","#ffee5f","#ffef63","#ffef67","#fff06a","#fff06e","#fff172","#fff177","#fff17b","#fff280","#fff285","#fff28a","#fff38e","#fff492","#fff496","#fff49a","#fff59e","#fff5a2","#fff5a6","#fff6aa","#fff6af","#fff7b3","#fff7b6","#fff8ba","#fff8bd","#fff8c1","#fff8c4","#fff9c7","#fff9ca","#fff9cd","#fffad1","#fffad4","#fffbd8","#fffcdb","#fffcdf","#fffde2","#fffde5","#fffde8","#fffeeb","#fffeee","#fffef1","#fffef4","#fffff6"]
for u in range(len(iron_palette)):
	color = tuple(int(iron_palette[u].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
	iron_palette[u] = (color[2],color[1],color[0])


upperbound_shift = 0
lowerbound_shift = 0
upperBound = 31
lowerBound = 25
upperBounds = collections.deque(maxlen=10)
lowerBounds = collections.deque(maxlen=10)

med_activate = True
med_filtersize = 3
med_filterwindow = collections.deque(maxlen=med_filtersize)
middleIndex = int(med_filtersize/2)

ma_activate = True
ma_filtersize = 2
ma_filterwindow = collections.deque(maxlen=ma_filtersize)


display_size = 480
interp_size = 64

mat = np.zeros((8,8,3), np.uint8)
matBW = np.zeros((8,8,1), np.uint8)
allBlack = np.zeros((48,64,1), np.uint8)

def inputHandle():
	global med_filtersize
	global ma_filtersize
	global ma_filterwindow
	global med_filterwindow
	global middleIndex
	global end
	global upperbound_shift
	global lowerbound_shift
	global crop
	global record
	while(end!=True):
		try:
			chars = input()
			if(chars[0] == 'm' and chars[1]=='a'):
				ma_filtersize = int(chars[2])
				print('ma_filtersize: %i' % ma_filtersize)
				ma_filterwindow = collections.deque(maxlen=ma_filtersize)
			elif(chars[0] == 'm' and chars[1]=='e' and chars[2]=='d'):
				med_filtersize = int(chars[3])
				print('med_filtersize: %i' % med_filtersize)
				med_filterwindow = collections.deque(maxlen=med_filtersize)
				middleIndex = int(med_filtersize/2)
			elif(chars == 'end'):
				end = True
			elif(chars[0]=='u' and chars[1]=='p'):
				upperbound_shift = float(chars[2:5])/100*-1
			elif(chars[0]=='l' and chars[1]=='o'):
				upperbound_shift = float(chars[2:5])/100
			elif(chars=='record'):
				print("START RECORDING")
				record = True
			elif(chars=='stop'):
				record = False
			else:
				print("INPUT ERROR")

		except:
			print("INPUT ERROR")
	

def median(med_filterwindow):
	filteredFrame = []
	for i, x in enumerate(med_filterwindow[0]):
		sort = []
		for t in range(med_filtersize):
			sort.append(med_filterwindow[t][i])
		sort.sort()
		filteredFrame.append(sort[middleIndex])
	return(filteredFrame)

def ma(ma_filterwindow):
	return(np.mean(ma_filterwindow,axis=0))

def draw(temp_values):
	for y in range(8):
		for x in range(8):
			cv2.rectangle(mat, (int(x), int(y)), (int((x+1)), int((y+1))), iron_palette[int(temp_values[y*8+(7-x)]*432)], -1)

def drawBW(temp_values):
	for y in range(8):
		for x in range(8):
			cv2.rectangle(matBW, (int(x), int(y)), (int((x+1)), int((y+1))), int(temp_values[y*8+(7-x)]*255), -1)
			


def scale(temp):
	if(temp<lowerBound):
		return(0)
	elif(temp>upperBound):
		return(1)
	else:
		try: 
			return((temp-lowerBound)/(upperBound-lowerBound))
		except:
			return(0)


cvFlag = False

t = time.time()

files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]


for filename in files:
	if(filename!='.DS_Store'):
		if(write==True):
			try:
				out_dir = os.path.join(mypath,'absrender_%s' % filename[0:13])
				os.mkdir(out_dir)
				print('processing: %s' % filename)
				cvFlag = True
			except:
				print('oops')
				#print('FILE EXIST: ','./grideye_dataset/x64/'+files[i][0:13])
				#cvFlag = False

		temp_frames = np.load(os.path.join(mypath,filename))
		med_filterwindow = collections.deque(maxlen=med_filtersize)
		ma_filterwindow = collections.deque(maxlen=ma_filtersize)

		for u in range(len(temp_frames)):

			temp_values = temp_frames[u]

			#filtering
			if(med_activate==True):
				med_filterwindow.append(temp_values)
				if(len(med_filterwindow)>=med_filtersize):
					temp_values = median(med_filterwindow)
	
			if(ma_activate==True):
				ma_filterwindow.append(temp_values)
				if(len(ma_filterwindow)>=ma_filtersize):
					temp_values = ma(ma_filterwindow)

			'''#scaling
			upperBounds.append(np.max(temp_values))
			if(len(upperBounds)>=5):
				upperBound = np.mean(upperBounds)+upperbound_shift

			lowerBounds.append(np.min(temp_values))
			if(len(lowerBounds)>=5):
				lowerBound = np.mean(lowerBounds)+lowerbound_shift
			'''
			temp_values = np.array(list(map(scale, temp_values)))

			#visualization
			#draw(temp_values)
			#resized = cv2.resize(mat, (interp_size,interp_size), interpolation=cv2.INTER_CUBIC)

			drawBW(temp_values)
			x = cv2.resize(matBW, (interp_size,interp_size), interpolation=cv2.INTER_CUBIC)
			x = x[int(interp_size*0.125):int(interp_size*0.875),:]
			cv2.imwrite(os.path.join(out_dir,'%i.jpg' % u),x)
			
			#cv2.imshow('img',x)
			#print(os.path.join('./grideye_dataset/yes_human/rgb/1632343831043','%i.jpg' % u))
			#cv2.imshow('rgb',cv2.imread(os.path.join('./grideye_dataset/yes_human/rgb/1632343831043','%i.jpg' % u)))
			#cv2.imshow('mask',cv2.imread(os.path.join('./grideye_dataset/yes_human/rgb/results_1632343831043','%i.jpg' % u)))
			#cv2.waitKey(0)
			#x = x[int(interp_size*0.125):int(interp_size*0.875),:]
			#if(write==True and cvFlag==True):
				#cv2.imwrite('./grideye_dataset/x64/'+files[i][0:13]+'/'+str(u)+'.jpg',x)
				#cv2.imwrite('./grideye_dataset/y64/'+files[i][0:13]+'/'+str(u)+'.jpg.jpg',allBlack)


			#print(x.shape)
			#grideyeDisplay = cv2.resize(resized, (display_size,display_size), interpolation=cv2.INTER_NEAREST)
			#grideyeDisplay = grideyeDisplay[int(display_size*0.125):int(display_size*0.875),:]


			#rgbDisplay = cv2.imread('/Users/shengyang/Documents/grideye_dataset/rgb/'+files[i][0:13]+'/'+str(u)+'.jpg')


			#overlay = cv2.addWeighted(rgbDisplay,0.4,grideyeDisplay,0.5,20)


			#cv2.imshow("grideye", x)
			#cv2.imshow("rgb", rgbDisplay)
			#cv2.imshow("overlay", overlay)
			#cv2.imshow("grideyeDisplay", grideyeDisplay)
			#if cv2.waitKey(1000000) & 0xFF == ord('q'):
			#	break
				












