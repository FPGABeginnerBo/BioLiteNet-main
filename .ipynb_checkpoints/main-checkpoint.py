# -*- coding: utf-8 -*-

import os
from functions import loop_train_test
from data import image_size_dict as dims
from data import draw_false_color, draw_gt, draw_bar

# remove abundant output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


## global constants
verbose = 1 # whether or not print redundant info (1 if and only if in debug mode, 0 in run mode)
run_times = 10#论文是10 # random run times, recommend at least 10
output_map = True # whether or not output classification map
only_draw_label = True # whether or not only predict labeled samples
disjoint = False # whether or not train and test on spatially disjoint samples

lr = 1e-3 # init learing rate
decay = 1e-3 # exponential learning rate decay
ws = 19 # window size
epochs = 64 # epoch
batch_size = 32   # batch size
model_type = 'demo'  # model type

def pavia_university_experiment():
    hp = {
        'pc': dims['1'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    # num_list = [66,186,21,31,13,50,13,37,9]#10%
    num_list = [132,372,42,62,26,100,26,74,18]  # 20%训练样本
    loop_train_test(dataID=1, num_list=num_list, verbose=verbose, run_times=run_times,
                    hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)

def indian_pine_experiment():
    hp = {
        'pc': dims['2'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    # num_list = [2, 71, 41, 12, 24, 36, 2, 24, 1, 48, 123, 29, 10, 63, 19, 4] # 5%
    # num_list = [40, 1420, 820, 230, 480, 720, 20, 470, 10, 970, 2400, 581, 200, 1250, 370, 85]  # 99%%
    # num_list = [4, 142, 82, 23, 48, 72, 2, 47, 1, 97, 24, 58, 20, 125, 37, 8]  # temp
    #num_list = [13, 473, 270, 73, 160, 240, 7, 153, 3, 323, 800, 194, 67, 416, 123, 28]  # 30%
    # num_list = [10, 355, 205, 57, 120, 180, 5, 117, 2, 282, 600, 140, 50, 312, 92, 21]  # 25%%
    # num_list = [4, 142, 82, 23, 48, 72, 2, 47, 1, 97, 24, 58, 20, 125, 37, 8]
    # num_list = [3, 95, 55, 15, 32, 48, 1, 31, 1, 64, 163, 39, 13, 84, 25, 6]  #划分15份

    # num_list = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]  # 0.1%训练样本

    num_list = [5, 143, 83, 24, 48, 73, 3, 48, 2, 97, 246, 59, 21, 127, 39, 9]  # 10%

    loop_train_test(dataID=2, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)

def Salinas_experiment():
    hp = {
        'pc': dims['3'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    num_list = [20, 37, 19, 13, 26, 39, 35, 112, 62, 32, 10, 19, 9, 10, 72, 18]
    loop_train_test(dataID=4, num_list=num_list, verbose=verbose, run_times=run_times,
                    hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)


def WHU_Hi_HanChuan_experiment():
    hp = {
        'pc': dims['5'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }


    # num_list =   [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7, 8] #0.1%训练样本
    # num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9] #0.5%训练样本
    num_list = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]  # 100训练样本
    # num_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] #0.1%训练样本
    #num_list = [50] * 15  #每类固定使用50个训练样本
    loop_train_test(dataID=5, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def WHU_Hi_HongHu_experiment():
    hp = {
        'pc': dims['6'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }


    # num_list =   [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7, 8] #0.1%训练样本
    #num_list = [2, 71, 41, 11, 24, 36, 2, 23, 2, 48, 122, 29, 10, 63, 19, 4] # 5%
    #num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9] #0.5%训练样本
    #num_list = [20, 37, 19, 13, 26, 39, 35, 112, 62, 32, 10, 19, 9, 10, 72, 18]  # 1%训练样本
    # num_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] #0.1%训练样本
    #num_list = [50] * 15  #每类固定使用50个训练样本
    #num_list = [25] * 22  #每类固定使用25个训练样本
    num_list = [50] * 22  #每类固定使用50个训练样本
    loop_train_test(dataID=6, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def WHU_Hi_LongKow_experiment():
    hp = {
        'pc': dims['7'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }


    # num_list = [345, 84, 30, 632, 42, 119, 671, 71, 52]  # 1%训练样本
    # num_list=[35,8,3,63,4,12,67,7,5]#0.1%
    num_list = [172,42,15,316,21,59,335,35,26] #0.5%

    loop_train_test(dataID=7, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def houston_university_experiment():
    hp = {
        'pc': dims['3'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
     # num_list =   [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7, 8] #0.1%训练样本
    #num_list = [2, 71, 41, 11, 24, 36, 2, 23, 2, 48, 122, 29, 10, 63, 19, 4] # 5%
    #num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9] #0.5%训练样本
    #num_list = [20, 37, 19, 13, 26, 39, 35, 112, 62, 32, 10, 19, 9, 10, 72, 18]  # 1%训练样本
    # num_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] #0.1%训练样本
    num_list = [50] * 15  #每类固定使用50个训练样本
    loop_train_test(dataID=3, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)



#实验
# pavia_university_experiment()
# indian_pine_experiment()
#houston_university_experiment()
# Salinas_experiment()
#WHU_Hi_HanChuan_experiment()
#WHU_Hi_HongHu_experiment()
WHU_Hi_LongKow_experiment()

# draw_false_color(dataID=1)
# draw_false_color(dataID=2)
#draw_false_color(dataID=3)
# draw_false_color(dataID=4)
#draw_false_color(dataID=5)
#draw_false_color(dataID=6)
draw_false_color(dataID=7)

#
# draw_bar(dataID=1)
# draw_bar(dataID=2)
#draw_bar(dataID=3)

# draw_gt(dataID=1, fixed=disjoint)
# draw_gt(dataID=2, fixed=disjoint)
# draw_gt(dataID=3, fixed=disjoint)

