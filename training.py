import os
import cv2
import time
import yaml
import argparse
import importlib
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
#import prepareData


def logger(loss_dict,filename=None,head_str=None):
    
    if not os.path.isfile(filename):
        atr = ''
        for key in loss_dict:
            atr += key+','
        with open(filename, "a") as f:
            f.write(atr+'\n')
            
    log_str = head_str+','
    prt_str = head_str+': '
    for key in loss_dict:
        log_str += '{:.2e},'.format(loss_dict[key])
        prt_str += '%s:%.4f, ' % (key,loss_dict[key])
    
    print(prt_str,end='\r')
    with open(filename, "a") as f:
        f.write(log_str+'\n')

def vali_logger(loss_dict,filename=None,head_str=None):
    
    if not os.path.isfile(filename):
        atr = ''
        for key in loss_dict:
            atr += key+','
        with open(filename, "a") as f:
            f.write(atr+'\n')
            
    log_str = head_str+','
    #prt_str = head_str+': '
    for key in loss_dict:
        log_str += '{:.2e},'.format(loss_dict[key])
        #prt_str += '%s:%.4f, ' % (key,loss_dict[key])
    
    #print(prt_str,end='\r')
    with open(filename, "a") as f:
        f.write(log_str+'\n')

early_stop = False
get_epoch = -1
#last_worst_loss_epoch = 0
gap_epoch = 10
loss_list = []
def check_progress(loss_value):
    loss_list.append(loss_value)
    print('loss_list:',loss_list)
    print('get_epoch:',get_epoch)
    print('loss_list.index(min(loss_list))',loss_list.index(min(loss_list)))
    print('minus',get_epoch-loss_list.index(min(loss_list)))
    if loss_value>min(loss_list) and get_epoch-loss_list.index(min(loss_list))>=gap_epoch:
        early_stop = True

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
@tf.function
def train_step(X,Y):
    #print('!!!!y_class_batch shape',Y[0].shape)
    #CHECK DATA SIZE
    if X.shape[0]!=BATCH_SIZE or Y[0].shape[0]!=BATCH_SIZE or Y[1].shape[0]!=BATCH_SIZE:
        raise ValueError('data length incorrect') 

    #FORWARD PASS AND RECORD GRADIENT
    with tf.GradientTape() as tape:
        class_pred, mask_pred = model(X, training=True)
        #print('shape class_pred: ',class_pred.shape)
        #print('shape Y[0]: ',Y[0].shape)
        class_loss = bce(Y[0], class_pred)
        mask_loss = tf.reduce_mean((mask_pred - Y[1])**2)
        loss = class_loss*cfg['loss_weights']['CLASS_LOSS'] + mask_loss*cfg['loss_weights']['MASK_LOSS'] 
          
    #APPLY GRADIENT
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables)) 
    losses = {'TOTAL_LOSS':loss, 'CLASS_LOSS':class_loss, 'MASK_LOSS':mask_loss}
    info = {'CLASS_PRED':class_pred, 'MASK_PRED':mask_pred}
    return losses, info


#READ CFG
parser = argparse.ArgumentParser()
parser.add_argument("cfg_path", type=str, help="path to cfg file")
args = parser.parse_args()
with open(args.cfg_path, "r") as stream:
    cfg = yaml.load(stream)
BATCH_SIZE = cfg['train']['BATCH_SIZE']


#GET MODEL AND OPTIMIZER
model = importlib.import_module(cfg['model']).get_model()
model.summary()
optimizer = keras.optimizers.Adam(learning_rate=cfg['optimizer']['LR'], beta_1=cfg['optimizer']['B1'], beta_2=cfg['optimizer']['B2'])


#MAKE LOG FOLDER
folder_name = os.path.join('./experiments',cfg['filename'])
#folder_name = os.path.join('./Downloads',cfg['filename'])
try:
    #os.mkdir(folder_name)
    os.mkdir(folder_name)
except:
    folder_name += str(int(time.time()))[-5:]
    os.mkdir(folder_name)

sample_folder = os.path.join(folder_name,'sample')
os.mkdir(sample_folder)
weight_folder = os.path.join(folder_name,'weight')
os.mkdir(weight_folder)


#WRITE CFG
with open(os.path.join(folder_name,'cfg.yaml'), 'a') as file:
    yaml.dump(cfg, file)


#GET DATASET
#x, y_mask, y_class = prepareData.get_data()
x = np.load('X.npy')
y_mask = np.load('y_mask.npy')
y_class = np.load('y_class.npy')
x, y_mask, y_class = shuffle(x, y_mask, y_class, random_state=0)
 
#np.save('X.npy',x)
#np.save('y_mask.npy',y_mask)
#np.save('y_class.npy',y_class)

#slice for vali here
x_vali, y_class_vali = x[:2000], y_class[:2000]
x, y_mask, y_class = x[2000:], y_mask[2000:], y_class[2000:]

#VALIDATION
def call_validation():
    #call model here and input vali data
    #y_class_vali
    y_class_pred = model.predict(x_vali)
    #print(y_class_pred[0].shape)
    #print(y_class_pred[0][0])
    #calculate acc? and class loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    bce1 = bce(y_class_vali, y_class_pred[0]).numpy()
    #print(bce1)
    acc = tf.keras.metrics.BinaryAccuracy()
    acc.update_state(y_class_vali, y_class_pred[0])
    acc1 = acc.result().numpy()
    #print('accuracy:',acc1)
    #go for dict: losses = {'TOTAL_LOSS':loss, 'CLASS_LOSS':class_loss, 'MASK_LOSS':mask_loss}
    validation = {'vali_BCE':bce1, 'vali_ACCURACY':acc1}
    vali_logger(validation, filename=os.path.join(folder_name,"vali.txt"), head_str="[E%i]"%epoch)

    check_progress(bce1)

#TRAIN
current_best = None
for epoch in range(cfg['train']['EPOCHS']):
    get_epoch = epoch
    for step in range(x.shape[0]//BATCH_SIZE):
        #GET DATA FOR CURRENT BATCH
        x_batch = x[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        y_mask_batch = y_mask[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        y_class_batch = y_class[step*BATCH_SIZE:(step+1)*BATCH_SIZE]

        #TRAIN
        #print('y_class shape', y_class.shape)
        #print('y_class_batch shape', y_class_batch.shape)
        losses, info = train_step(x_batch,[y_class_batch,y_mask_batch])

        #LOG
        logger(losses, filename=os.path.join(folder_name,"log.txt"), head_str="[E%iS%i]"%(epoch,step))
    
    #SAVE MODEL
    if epoch%cfg['train']['CHECKPOINT_FREQ'] == 0:
        if current_best==None or losses['TOTAL_LOSS']<current_best:
            current_best = losses['TOTAL_LOSS']
            model.save(os.path.join(weight_folder, 'E%iS%i_%.4f.h5'%(epoch, step, losses['TOTAL_LOSS'])))

    #SAVE SAMPLES
    if epoch%cfg['train']['SAMPLE_FREQ'] == 0:
        vis_h1 = tf.repeat(x_batch[0][:,:,0:1], 3, axis=-1)#ABS or SCA choose your self
        vis_h2 = tf.repeat(info['MASK_PRED'][0], 3, axis=-1).numpy()
        vis_h2 = cv2.putText(vis_h2, '%.1f'%info['CLASS_PRED'][0].numpy(), (0,48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (-1,1,-1), 1, cv2.LINE_AA)
        vis_h3 = tf.repeat(y_mask_batch[0], 3, axis=-1).numpy()
        vis_h3 = cv2.putText(vis_h3, '%.1f'%y_class_batch[0][0], (0,48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (-1,-1,1), 1, cv2.LINE_AA)
        for i in range(1,10,1):
            h1_img = tf.repeat(x_batch[i][:,:,0:1],3,axis=-1)
            h2_img = tf.repeat(info['MASK_PRED'][i], 3, axis=-1).numpy()
            h2_img = cv2.putText(h2_img, '%.1f'%info['CLASS_PRED'][i].numpy(), (0,48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (-1,1,-1), 1, cv2.LINE_AA)
            h3_img = tf.repeat(y_mask_batch[i], 3, axis=-1).numpy()
            h3_img = cv2.putText(h3_img, '%.1f'%y_class_batch[i][0], (0,48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (-1,-1,1), 1, cv2.LINE_AA)
            vis_h1 = np.concatenate((vis_h1, h1_img), axis=1)
            vis_h2 = np.concatenate((vis_h2, h2_img), axis=1)
            vis_h3 = np.concatenate((vis_h3, h3_img), axis=1)
        #print(vis_h1.shape)
        #print(vis_h2.shape)
        #print(vis_h3.shape)
        vis = np.concatenate(([vis_h1, vis_h2, vis_h3]), axis=0)
        cv2.imwrite(os.path.join(sample_folder,'E%iS%i_sample.png' % (epoch,step)), (vis+1)/2*255)

    call_validation()
    if early_stop==True:
        break



