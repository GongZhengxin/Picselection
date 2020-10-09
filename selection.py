import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
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


def get_label_map(data_path='/nfs/e3/ImgDatabase/ImageNet_2012', train_fold='ILSVRC2012_img_train',
                  info_fold='label', label_file='synset_words.txt'):
    # get label mapping dict
    f_word = open(pjoin(data_path, info_fold, label_file))
    class_set = f_word.readlines()
    # label mapping dict
    label_map = {}
    for line in class_set:
        key, value = line.split(' ', 1)[0], line.split(' ', 1)[1]
        label_map[key] = value.split(',')[0].replace('\n', '')
    f_word.close()
    return label_map


def get_name(index, string):
    """
    :param index: int
           string: str
    :return: file_name [str]
    """
    # prep for create file names
    if index < 10:
        file_name = 'class_00' + str(index) + string
    elif index < 100:
        file_name = 'class_0' + str(index) + string
    else:
        file_name = 'class_' + str(index) + string
    return file_name


def get_index(target, mode='number', match=False):
    label_map = get_label_map()
    target = str(target)
    index = 'Not find'
    if mode == 'number':
        arr_labels = np.array(list(label_map.keys())).astype(np.str)
        index = np.argwhere(arr_labels == target).squeeze()
    elif mode == 'name':
        arr_labels = np.array(list(label_map.values())).astype(np.str)
        index = np.argwhere(arr_labels == target).squeeze()
        if index.size == 0 & match == False:
            target = target.lower()
            index = np.argwhere(arr_labels == target).squeeze()
            if index.size == 0:
                target = target.capitalize()
                index = np.argwhere(arr_labels == target).squeeze()
    return index


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

label_map = get_label_map(data_path='/nfs/e3/ImgDatabase/ImageNet_2012', train_fold='ILSVRC2012_img_train',
                          info_fold='label', label_file='synset_words.txt')

# # check point
# print('lines read over')

# get activation & soft max value, save the file out
# the whole train-set load into a DataFrame
df_train = pd.read_csv(pjoin(data_path, info_fold, train_file), sep=' ', header=None)
df_train.columns = ['path', 'label']

# loop to get all activations of any images in the class
for index in range(249, 1000):
    # prep for create file names
    file_name = get_name(index, '.stim.csv')
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
            wh_list.append(resolution[0] / resolution[1])
        # set the filter columns
        df_train_class.insert(1, 'width', w_list)
        df_train_class.insert(2, 'height', h_list)
        df_train_class.insert(3, 'w:h', wh_list)
        del w_list, h_list, wh_list

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
        del class_path, label_number, class_name

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
        col_name = 'act' + str(index)
        df_train_class[col_name] = df_act[col_name]
        del activation
        # # check point
        # print(df_train_class.iloc[:5, :5])

        # save out
        df_train_class.to_csv(file_path.replace('.stim', ''), index=None)
        print('save out!')

t2 = time.time()
print('Time consuming: ', t2 - t1)
print('-------over----------')


# =========filtering=============

# # filtered_num = list()
# for index in range(1000):
#
#     file = get_name(index, '.csv')
#     if not os.path.exists(pjoin('OldStimulusFiles', file)):
#         continue
#     df_temp = pd.read_csv(pjoin('OldStimulusFiles', file))
#     # w-h ratio filter
#     wh_filter = np.abs((np.array(df_temp['w:h']) - df_temp['w:h'].mean()) / df_temp['w:h'].std()) < 3
#     # soft max filter
#     arr_sftmx = np.array(df_temp['softmax'])
#     sm_filter = arr_sftmx >= np.sort(arr_sftmx)[int(0.1 * len(arr_sftmx))]
#     # whole filter
#     filter_list = [int(i) for i in wh_filter & sm_filter]
#     # filtered_num.append(np.sum(filter_list))
#     # add new column
#     df_temp['filter'] = filter_list
#     # refresh file
#     df_temp.to_csv(pjoin('StimulusFiles', 'class_000.csv'))
# del arr_sftmx, df_temp, wh_filter, sm_filter


# =========filtering & CAM property==========
def get_mask(img, threshold: float):
    """
    generate the mask for a heatmap where value become 1 for
    which larger than threshold, else place 0

    :param img:
    :param threshold: float
    :return:
    """
    mask = np.array(img)
    mask[np.where(stats.zscore(mask, axis=None) <= threshold)] = 0
    mask[np.where(stats.zscore(mask, axis=None) > threshold)] = 1
    return mask


def get_mean_coordinate(mask):
    coordinates = np.where(mask == 1)
    x = (coordinates[1].mean() - mask.shape[1] / 2) / mask.shape[1]
    y = (coordinates[0].mean() - mask.shape[0] / 2) / mask.shape[0]
    return np.array((x, y), dtype=np.float16)


# (coordinates[0].mean(), coordinates[1].mean())

t1 = time.time()
hist_dict = dict()
for index in range(1000):
    print(index)
    # load .npy
    cam_filename = get_name(index, '.npy')
    cam_file = pjoin('CAM', 'heatmap', cam_filename)
    if os.path.exists(cam_file):
        channel_maps = np.load(cam_file)
    else:
        continue
    # .csv file
    csv_filename = get_name(index, '.csv')
    csv_file = pjoin('StimulusFiles', csv_filename)
    df_temp = pd.read_csv(csv_file)
    # w-h ratio filter
    wh_filter = np.abs((np.array(df_temp['w:h']) - df_temp['w:h'].mean()) / df_temp['w:h'].std()) < 3
    # soft max filter
    arr_sftmx = np.array(df_temp['softmax'])
    sm_filter = arr_sftmx >= np.sort(arr_sftmx)[int(0.1 * len(arr_sftmx))]
    # whole filter
    filter_list = [int(i) for i in wh_filter & sm_filter]
    # filtered_num.append(np.sum(filter_list))
    # add new column
    df_temp['filter'] = filter_list
    # print('filtered!')
    # CAM propeties
    ratios, center_xs, center_ys = list(), list(), list()
    for pic in range(len(df_temp)):
        w, h = df_temp.loc[pic, 'width'], df_temp.loc[pic, 'height']
        cur_map = channel_maps[pic]
        # because 'width' derived from np.shape[0],
        # and it correspond to height of Image. So
        # does 'height' do.
        heatmap = cv2.resize(cur_map, (h, w))
        heatmap_mask = get_mask(heatmap, threshold=0.6)
        ratio = np.mean(heatmap_mask)
        center_x, center_y = get_mean_coordinate(heatmap_mask)[0], get_mean_coordinate(heatmap_mask)[1]
        ratios.append(ratio)
        center_xs.append(center_x)
        center_ys.append(center_y)
    df_temp['ratio'] = ratios
    df_temp['x'] = center_xs
    df_temp['y'] = center_ys
    df_temp.to_csv(pjoin('StimulusFiles', csv_filename))
    # print('refreshed!')
    # make histgram
    df_filtered = df_temp[df_temp['filter'] == 1].reset_index(drop=True)
    df_filtered.to_csv(pjoin('FilteredStimFiles', csv_filename))
    del df_temp
    # print('New file!')

    act = 'act' + str(index)
    actives, ratios = np.array(df_filtered[act]), np.array(df_filtered['ratio'])
    center_xs, center_ys = np.array(df_filtered['x']), np.array(df_filtered['y'])
    hist_actives, hist_ratios = np.histogram(actives, bins=40), np.histogram(ratios, bins=40)
    hist_pos = np.histogram2d(center_xs, center_ys, bins=10)
    del actives, ratios, center_xs, center_ys

    hist_dict[csv_filename] = {'act': (hist_actives[0], (hist_actives[1][0], hist_actives[1][-1])),
                               'ratio': (hist_ratios[0], (hist_ratios[1][0], hist_ratios[1][-1])),
                               'position': (hist_pos[0], np.array([[hist_pos[1][0], hist_pos[1][-1]],
                                                                   [hist_pos[2][0], hist_pos[2][-1]]]))
                               }
    # print('histogramed!')
with open('histogram.pkl', 'wb') as f:
    pickle.dump(hist_dict, f)
# print('saved!')
t2 = time.time()
print('consumed:', t2 - t1)


# =============random selector================
def hist_MSE(population, sample):
    return ((population / population.sum() - sample / sample.sum()) ** 2).mean()


def print_r(a, act, ratio, pos):
    if type(a) == tuple:
        a = int(a[0])
    print('act:', act[a], 'ratio:', ratio[a], 'pos:', pos[a])


t1 = time.time()
selected_images = dict()
for index in range(1, 1000):
    # population hist properties
    file_name = get_name(index, '.csv')
    population_hist = hist_dict[file_name]
    pop_hist_act, pop_hist_ratio = population_hist['act'], population_hist['ratio']
    pop_hist_pos = population_hist['position']
    range_act, range_ratio, range_pos = pop_hist_act[1], pop_hist_ratio[1], pop_hist_pos[1]
    # population
    df_temp = pd.read_csv(pjoin('FilteredStimFiles', file_name))
    act = 'act' + str(index)
    arr_active = np.array(df_temp[act])
    arr_ratio, arr_xs, arr_ys = np.array(df_temp['ratio']), np.array(df_temp['x']), np.array(df_temp['y'])
    # random selection & random properties
    lists = dict()
    # mse_act, mse_ratio, mse_pos = list(), list(), list()
    r_act, r_ratio, r_pos = list(), list(), list()
    for times in range(1000):
        np.random.seed()
        sample_list = np.random.choice(len(df_temp), 80, replace=False)
        lists[times] = sample_list
        # get parameter
        sample_act, sample_ratio = arr_active[sample_list], arr_ratio[sample_list]
        sample_xs, sample_ys = arr_xs[sample_list], arr_ys[sample_list]
        # histogram
        sam_hist_act, sam_hist_ratio = \
            np.histogram(sample_act, bins=40, range=range_act), np.histogram(sample_ratio, bins=40, range=range_ratio)
        sam_hist_pos = np.histogram2d(sample_xs, sample_ys, bins=10, range=range_pos)
        del sample_act, sample_ratio, sample_xs, sample_ys

        # # mse
        # mse_act.append(hist_MSE(pop_hist_act[0], sam_hist_act[0]))
        # mse_ratio.append(hist_MSE(pop_hist_ratio[0], sam_hist_ratio[0]))
        # mse_pos.append(hist_MSE(pop_hist_pos[0].flatten(), sam_hist_pos[0].flatten()))
        # pearson r
        r_act.append(stats.pearsonr(pop_hist_act[0], sam_hist_act[0])[0])
        r_ratio.append(stats.pearsonr(pop_hist_ratio[0], sam_hist_ratio[0])[0])
        r_pos.append(stats.pearsonr(pop_hist_pos[0].flatten(), sam_hist_pos[0].flatten())[0])
    sum = np.array(r_act) + np.array(r_ratio) + np.array(r_pos)
    # sum = np.array(mse_act) + np.array(mse_ratio) + np.array(mse_pos)
    max_index = int(np.argmax(sum))
    selected_images[index] = {'sample': np.array(df_temp.loc[lists[max_index], 'path']).astype(np.str),
                              'r': (r_act[max_index], r_ratio[max_index], r_pos[max_index])}
    sample_list = np.array(df_temp.loc[lists[max_index], 'path']).astype(np.str)
    for image in sample_list:
        old_file_path = pjoin('/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_train', image)
        #
        img = Image.open(old_file_path)
        img = img.resize((375, 375))
        fold, file = image.split('/')
        if not os.path.exists(pjoin('SelectedImages', fold)):
            os.mkdir(pjoin('SelectedImages', fold))
        new_file_path = pjoin('SelectedImages', fold, file)
        img.save(new_file_path)

    print('save all!')
    t2 = time.time()

with open('selectedImages.pkl', 'wb') as f:
    pickle.dump(selected_images, f)



# =============resize=============


class trash_code():
    # print('+==========+============+============+============+====================+') print('|class num |wh num,
    # ratio|sm num,ratio|&& num,ratio| name               |') print(
    # '+----------+------------+------------+------------+--------------------+') print('|{:10s}|{:4d}, {:.4f}|{:4d},
    # {:.4f}|{:4d}, {:.4f}|{:16s}|'.format(df_temp.loc[0, 'path'].split('/')[0], np.sum(wh_filter),
    # np.mean(wh_filter), np.sum(sm_filter), np.mean(sm_filter), np.sum(filter_list), np.mean(filter_list),
    # label_map[ df_temp.loc[0, 'path'].split('/')[0]])) print('{} remains {:.3f}: {}'.format(label_map[df_temp.loc[
    # 0,'path'].split('/')[0]], np.mean(filter_list), np.sum(filter_list)))

    def cv2_read_write(self):
        up_hm = cv2.applyColorMap(cv2.resize(cur_map, (h, w)), cv2.COLORMAP_JET)
        original_pic = cv2.imread(pjoin(data_path, train_fold, df_temp.loc[pic, 'path']))
        mixture = up_hm * 0.3 + original_pic * 0.5
        cv2.imwrite('example{}.jpg'.format(pic), mixture)

    def change_filename(self):
        path = 'CAM/heatmap'
        filename_list = os.listdir(path)
        for filename in filename_list:
            old_name = pjoin(path, filename)
            index = int(filename.split('_')[0].replace('class', ''))
            new_name = pjoin(path, get_name(index, '.npy'))
            os.rename(old_name, new_name)


def simulation(bins=20):
    np.random.seed(2020)
    population = np.random.randn(1000)
    hist_population = np.histogram(population, bins=bins)
    range = (hist_population[1][0], hist_population[1][-1])
    np.random.seed()
    sample = np.random.choice(population, 80)
    hist_sample = np.histogram(sample, range=range, bins=bins)
    plt.close('all')
    r = stats.pearsonr(hist_population[0], hist_sample[0])[0]
    return r


def simulation2d(bins=20):
    np.random.seed(2020)
    population_x = np.random.randn(1000)
    np.random.seed(200)
    population_y = np.random.randn(1000)

    hist_population = np.histogram2d(population_x, population_y, bins=bins)
    range = [[hist_population[1][0], hist_population[1][-1]], [hist_population[2][0], hist_population[2][-1]]]

    np.random.seed()
    sample = np.random.choice(1000, 80)
    hist_sample = np.histogram2d(population_x[sample], population_y[sample], range=range, bins=bins)
    plt.close('all')
    p = hist_population[0].flatten()
    s = hist_sample[0].flatten()
    r = stats.pearsonr(p, s)[0]
    return r


def simulate(times=100, simulation=simulation2d):
    t1 = time.time()
    bin_list = [5, 6, 10, 15, 20, 40]
    mean_r, std_r = list(), list()
    for bin in bin_list:
        rs = list()
        i = 0
        while i < times:
            i += 1
            rs.append(simulation(bins=bin))
        mean_r.append(np.mean(rs))
        std_r.append(np.std(rs))
    plt.plot(bin_list, mean_r)
    plt.errorbar(bin_list, mean_r, yerr=std_r)
    plt.show()
    t2 = time.time()
    print('consumed: ', t1 - t2)


def print_max_min(act, ratio, pos):
    print('act:', np.max(act), np.min(act))
    print('ratio:', np.max(ratio), np.min(ratio))
    print('pos:', np.max(pos), np.min(pos))



def visualize_hist(sample, arr_active, arr_ratio, arr_xs, arr_ys, notation='none'):
    fig = plt.figure(figsize=(8, 4))
    arr_all = (arr_active, arr_ratio, arr_xs, arr_ys)
    for i in [0, 1, 2, 3, 4, 5]:
        ax = fig.add_subplot(2, 3, i + 1)
        if i == 0 or i == 1:
            arr = arr_all[i]
            ax.hist(arr, bins=40)
        elif i == 3 or i == 4:
            arr = arr_all[i - 3][sample]
            range = (np.histogram(arr_all[i-3], bins=40)[1][0], np.histogram(arr_all[i-3], bins=40)[1][-1])
            ax.hist(arr, bins=40, range=range)
        elif i == 2 or i == 5:
            arr1, arr2 = arr_all[2], arr_all[3]
            if i < 3:
                ax.hist2d(arr1, arr2, bins=10)
            else:
                hist = np.histogram2d(arr1, arr2, bins=10)
                range = np.array([[hist[1][0], hist[1][-1]], [hist[2][0], hist[2][-1]]])
                ax.hist2d(arr1[sample], arr2[sample], bins=10, range=range)
    plt.xlabel(notation)
    plt.show()


# =============substitute=============
# collecting
main_fold = 'out'
substitude_dict = dict()
for i in range(1000):
    substitude_dict[i] = list()

for sub in range(20):
    fold_name = 'sub'+str(sub)
    for session in range(4):
        file_name = 'expImage_' + fold_name + '_session' + str(session) + '.txt'
        with open(pjoin(main_fold, fold_name, file_name)) as f:
            records = f.readlines()
            for i in range(len(records)):
                record = records[i].replace('\n', '').split('\t')
                if record[-2] == 0:
                    index = get_index(record[-1].split('/')[0])
                    substitude_dict[index].append(record[-1])

with open('selectedImages.pkl', 'rb') as f:
    selected_images = pickle.load(f)

for index in range(1000):
    filename = get_name(index, '.csv')
    df = pd.read_csv(pjoin('FilteredStimFiles', filename))
    stim_set = set(np.array(df['path']).astype(np.str)) - set(selected_images[index]['sample'])





# dir_list = os.listdir('SelectedImgaes')
# for fold in dir_list:
#     index = get_index(fold)
#     selected_images[index]


