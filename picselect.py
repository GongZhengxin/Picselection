import os
import cv2
import numpy as np
import pandas as pd

from os.path import join as pjoin
from dnnbrain.io.fileio import StimulusFile
from dnnbrain.dnn.core import Stimulus, Mask
from dnnbrain.dnn.models import Resnet152
from shutil import copy

import time

t1 = time.time()
# make dir for .stim.csv
if not os.path.exists('StimulusFiles'):
    os.mkdir('StimulusFiles')
if not os.path.exists('FilteredStimFiles'):
    os.mkdir('FilteredStimFiles')
if not os.path.exists('OriginalImages'):
    os.mkdir('OriginalImages')

# where label & pic fold locate
main_path = '/nfs/e3/ImgDatabase/ImageNet_2012'
train_fold = 'ILSVRC2012_img_train'
txt_fold = 'label'
path_file = 'train.txt'
word_file = 'synset_words.txt'
# resolution threshold
resol_thres = 375

select_num = 30

try:
    # file open
    df_train = pd.read_csv(pjoin(main_path, txt_fold, path_file), sep=' ', header=None)
    df_train.columns = ['path', 'label']
    f_word = open(pjoin(main_path, txt_fold, word_file))
    # print('file open, reading lines')
    # class index
    class_set = f_word.readlines()
    # print('lines read over')

    # class selection loop
    for index in range(20):
        # prep for create .stim.csv
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
            txt_path = file_path.replace('StimulusFiles', 'FilteredStimFiles').replace('.stim.csv', '.txt')
            with open(txt_path) as txtf:
                # print('there are',len(txtf.readlines()), 'images')
                for line in txtf.readlines():
                    line = line.replace('\n', '')
                    # move image stimulus to new fold
                    old_file_path = pjoin(main_path, train_fold, line)
                    # 
                    fold, file = line.split('/')
                    if not os.path.exists(pjoin('OriginalImages', fold)):
                        os.mkdir(pjoin('OriginalImages', fold))
                    new_file_path = pjoin(os.getcwd(), 'OriginalImages', fold, file)
                    copy(old_file_path, new_file_path)
            print(txt_path, ' copy done!')
            continue
        else:
            print('\nstarting class', index)
            # create .stim.csv
            stim_file = StimulusFile(file_path)
            # initilize the filter list
            filter_list = list()

            # select all image path of the class **index**
            df_class_path = pd.DataFrame(df_train.loc[df_train['label'] == index, 'path'])
            image_path_all = np.array(df_class_path['path']).astype(np.str)
            for row in range(len(df_class_path)):
                # get image path
                image = image_path_all[row]
                # obtain the width x heigth of image
                img_path = pjoin(main_path, train_fold, image)
                resolution = np.array(cv2.imread(img_path).shape)[0:2]
                # select images larger than threshold
                if np.min(resolution) >= resol_thres:
                    filter_list.append(1)
                else:
                    filter_list.append(0)
            #                    print(img_path)
            #                 # for fast check
            #                if row >=10:
            #                    filter_list.extend((len(df_class_path)-len(filter_list))*[0])
            #                    break
            # check wheter the list lenghth equals dataframe
            #            print('check ', len(filter_list)==len(df_class_path))
            # set the filter columns
            df_class_path.insert(1, 'filter', filter_list)
            # get filtered class path
            df_class_filtered = pd.DataFrame(df_class_path.loc[df_class_path['filter'] == 1, 'path'])

            # get class name
            _, class_name = class_set[index].split(' ', 1)
            # print primilinary selection number
            print(class_name[:5], ' has ', len(df_class_filtered), ' images\n')

            # construct data dict
            data = {}
            data['stimID'] = list(df_class_filtered['path'])
            # write .stim.csv
            stim_file.write('image', pjoin(main_path, train_fold), data,
                            class_index={'index': index}, class_name=class_name)

            ## generate activation
            # create stimulus
            stimuli = Stimulus()
            stimuli.load(file_path)
            # create mask for absolute value 
            dmask = Mask()
            layer = 'fc'
            dmask.set(layer, channels=[index])

            # extract activation
            dnn = Resnet152()
            activation = np.array(dnn.compute_activation(stimuli, dmask).get(layer))
            df_class_filtered.insert(1, 'activ', activation.squeeze())
            df_class_filtered.sort_values(by='activ', ascending=False)
            # get the sorted images path
            sorted_image_list = np.array(df_class_filtered['path']).astype(np.str)
            #            print('Selction starting')
            # get critical percentile & select
            pic_num = len(sorted_image_list)
            selected_image = list()
            for percent in [0.25, 0.5, 0.75, 1]:
                # create range
                q1, q2 = int((percent - 0.25) * pic_num), int(percent * pic_num)
                # random choose the top range pics
                selection = np.random.permutation(np.linspace(q1, q2, q2 - q1, endpoint=False))[:select_num]
                selection = selection.astype(np.int8)
                selected_image.extend(list(selection))
            # get the path
            final_selection = sorted_image_list[selected_image]
            # create selected .txt
            final_stimfile = file_name.replace('.stim.csv', '.txt')
            with open(pjoin('FilteredStimFiles', final_stimfile), 'w') as sf:
                for line in final_selection:
                    sf.write(line + '\n')
                    # move image stimulus to new fold
                    old_file_path = pjoin(main_path, train_fold, line)
                    # 
                    fold, _ = line.split('/')
                    new_file_path = pjoin(os.getcwd(), 'OriginalImages', fold)
                    copy(old_file_path, new_file_path)
finally:
    f_word.close()
    print('-------over----------')

t2 = time.time()
print('Time:', t2 - t1)


def check_copy_files(class_num):
    miss_dict = {}
    search_range = []
    if type(class_name) == int:
        search_range = range(class_name)
    elif type(class_num) == list:
        search_range = class_num
    for i in search_range:
        if i < 10:
            filtered_file = 'class_00' + str(i) + '.txt'
        elif i < 100:
            filtered_file = 'class_0' + str(i) + '.txt'
        else:
            filtered_file = 'class_' + str(i) + '.txt'
        with open(pjoin('FilteredStimFiles', filtered_file)) as sf:
            print('total image:', len(sf.readlines()))
            miss_images = []
            for oneline in sf.readlines():
                cur_fold, cur_image = oneline.split('/')
                dirlist = os.listdir(pjoin('OriginalImages', cur_fold))
                if not(cur_image in dirlist):
                    miss_images.append(line)
            print('miss image:', len(miss_images), '\n')
            miss_dict[filtered_file] = miss_images
    


