import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from dnnbrain.dnn.base import ip

from os.path import join as pjoin
from random import shuffle
from shutil import copy
from collections import OrderedDict
from matplotlib.ticker import NullFormatter
from dnnbrain.dnn.core import Stimulus, Mask
from dnnbrain.dnn.models import Resnet152
from dnnbrain.io.fileio import StimulusFile

t1 = time.time()
# check folds in current dir
if not os.path.exists('StimulusFiles'):
    os.mkdir('StimulusFiles')
if not os.path.exists('FilteredStimFiles'):
    os.mkdir('FilteredStimFiles')
if not os.path.exists('OriginalImages'):
    os.mkdir('OriginalImages')
if not os.path.exists('CroppedImages'):
    os.mkdir('CroppedImages')


def get_name(index):
    """
    :param index: int
    :return: file_name [str]
    """
    # prep for create file names
    if index < 10:
        file_name = 'class_00' + str(index) + '.stim.csv'
    elif index < 100:
        file_name = 'class_0' + str(index) + '.stim.csv'
    else:
        file_name = 'class_' + str(index) + '.stim.csv'
    return file_name


# where label & pic fold locate
data_path = '/nfs/e3/ImgDatabase/ImageNet_2012'
train_fold = 'ILSVRC2012_img_train'
info_fold = 'label'
train_file = 'train.txt'
label_file = 'synset_words.txt'
# resolution threshold
res_threshold = 375
# number of selection
select_num = 30

# get label mapping dict
f_word = open(pjoin(data_path, info_fold, label_file))
class_set = f_word.readlines()
# label mapping dict
label_map = {}
for line in class_set:
    key, value = line.split(' ', 1)[0], line.split(' ', 1)[1]
    label_map[key] = value.split(',')[0].replace('\n', '')
f_word.close()

# # check point
# print('lines read over')

# get activation & soft max value, save the file out
# the whole train-set load into a DataFrame
df_train = pd.read_csv(pjoin(data_path, info_fold, train_file), sep=' ', header=None)
df_train.columns = ['path', 'label']

# loop to get all activations of any images in the class
for index in range(249, 1000):
    # prep for create file names
    if index < 10:
        file_name = 'class_00' + str(index) + '.stim.csv'
    elif index < 100:
        file_name = 'class_0' + str(index) + '.stim.csv'
    else:
        file_name = 'class_' + str(index) + '.stim.csv'
    file_path = pjoin('StimulusFiles', file_name)
    # check file existence
    if os.path.exists(file_path):
        print(file_name, ' already exists, please mannually operate it\n')
        continue
    else:
        # check point
        print('\nstarting class', index)

        # initialize the width height & w:H list
        w_list, h_list, wh_list = list(), list(), list()
        # select all image path of the class **index**
        df_train_class = pd.DataFrame(df_train.loc[df_train['label'] == index, 'path']).reset_index(drop=True)

        # # ==== for fast check=====
        # df_train_class = df_train_class[:5]

        # transform to np.array for sorting
        all_image_path = np.array(df_train_class['path']).astype(np.str)
        for row in range(len(df_train_class)):
            # get image path
            image = all_image_path[row]
            # obtain the width x height of image
            img_path = pjoin(data_path, train_fold, image)
            resolution = np.array(cv2.imread(img_path).shape)[0:2]
            w_list.append(resolution[0])
            h_list.append(resolution[1])
            wh_list.append(resolution[0]/resolution[1])
        # set the filter columns
        df_train_class.insert(1, 'width', w_list)
        df_train_class.insert(2, 'height', h_list)
        df_train_class.insert(3, 'w:h', wh_list)

        # # check point
        # print(df_train_class[:10])

        # get class path list
        class_path = list(df_train_class['path'])
        # filtered class path:
        # list(df_train_class.loc[df_train_class['filter'] == 1, 'path'])

        # get class name
        label_number = class_path[0].split('/')[0]
        class_name = label_map[label_number]
        # print preliminary selection number
        # print(class_name[:5], ' has ', len(filtered_class_path), ' images\n')
        
        
        # construct data dict
        data = dict()
        data['stimID'] = class_path
        # create .stim.csv
        stim_file = StimulusFile(file_path)
        # write .stim.csv
        stim_file.write('image', pjoin(data_path, train_fold), data,
                        class_index={'index': index, 'name': class_name})

        # generate activation
        # create stimulus
        stimuli = Stimulus()
        stimuli.load(file_path)
        # create mask for absolute value
        dmask = Mask()
        layer = 'fc'
        # activation value
        dmask.set(layer, channels='all')

        # extract activation
        dnn = Resnet152()
        activation = np.array(dnn.compute_activation(stimuli, dmask).get(layer)).astype(np.float32)
        activation = activation.squeeze()

        # # delete .stim.csv file
        # os.remove(file_path)

        # transform activation data
        df_act = pd.DataFrame(activation)
        if len(activation.shape) == 1:
            col_num = 1
        else:
            col_num = activation.shape[-1]
        df_act.columns = ['act' + str(i) for i in range(col_num)]

        # calculate soft-max value
        df_train_class['softmax'] = np.exp(df_act['act' + str(index)]) / np.sum(
            np.exp(df_act.loc[:, ['act' + str(j) for j in range(col_num)]]),
            axis=1)
        # get activation value
        col_name = 'act'+str(index)
        df_train_class[col_name] = df_act[col_name]

        # # check point
        # print(df_train_class.iloc[:5, :5])

        # save out
        df_train_class.to_csv(file_path.replace('.stim', ''), index=None)
        print('save out!')

t2 = time.time()
print('Time consuming: ', t2-t1)
print('-------over----------')

# =========filtering=============
for i in range(1000):
    file = 'class_000.csv'
    df_temp = pd.read_csv(pjoin('StimulusFiles', file))
    # w-h ratio filter
    wh_filter = np.abs((np.array(df_temp['w:h'])-df_temp['w:h'].mean())/df_temp['w:h'].std()) < 3
    # soft max filter
    sm_filter = np.array(df_temp['softmax']) >= 0.9
    # whole filter
    filter_list = [int(i) for i in wh_filter & sm_filter]
    # add new column
    df_temp['filter'] = filter_list
    # refresh file
    df_temp.to_csv(pjoin('StimulusFiles', 'class_000.csv'))

# =============random selector================
select_num = 4
csvfiles = ['class_00'+str(i)+'.csv' if i < 10 else 'class_0'+str(i)+'.csv' for i in range(30)]
#
file = 'class_000.csv'
df_class = pd.read_csv(pjoin('StimulusFiles', file))
df_train_class = df_class.loc[df_class['filter'] == 1, ['path', 'act0']].reset_index(drop=True)
df_train_class.sort_values(by='act0', ascending=False)
# get the sorted images path
sorted_image_list = np.array(df_train_class['path']).astype(np.str)
#            print('Selction starting')
# get critical percentile & select
pic_num = len(sorted_image_list)
selected_image = list()
percents = np.linspace(0, 1, 21)[1:]
for percent in percents:
    # create range
    q1, q2 = int((percent - 0.05) * pic_num), int(percent * pic_num)
    # random choose the top range pics
    selection = [i for i in range(q1, q2)]
    shuffle(selection)
    selected_image.extend(selection[:select_num])
# get the path
final_selection = sorted_image_list[selected_image]
# create selected .txt
final_stimfile = file.replace('.csv', '.txt')
main_path = '/nfs/e3/ImgDatabase/ImageNet_2012'
train_fold = 'ILSVRC2012_img_train'
with open(pjoin('FilteredStimFiles', final_stimfile), 'w') as sf:
    for line in final_selection:
        sf.write(line + '\n')
        # move image stimulus to new fold
        old_file_path = pjoin(main_path, train_fold, line)
        #
        fold, file = line.split('/')
        if not os.path.exists(pjoin('OriginalImages', fold)):
            os.mkdir(pjoin('OriginalImages', fold))
        new_file_path = pjoin(os.getcwd(), 'OriginalImages', fold, file)
        copy(old_file_path, new_file_path)


# =============crop/resize=============