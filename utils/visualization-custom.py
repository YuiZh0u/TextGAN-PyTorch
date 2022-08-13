# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : visualization.py
# @Time         : Created at 2019-03-19
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import matplotlib.pyplot as plt
import numpy as np

title_dict = {
    'gen_pre_loss': 'pre_loss',
    'gen_adv_loss': 'g_loss',
    'gen_mana_loss': 'mana_loss',
    'gen_work_loss': 'work_loss',
    'dis_loss': 'd_loss',
    'dis_train_acc': 'train_acc',
    'dis_eval_acc': 'eval_acc',
    'NLL_oracle': 'NLL_oracle',
    'NLL_gen': 'NLL_gen',
    'BLEU-3': 'BLEU-3',
    'BLEU': 'BLEU',
}

color_list = ['#e74c3c', '#e67e22', '#f1c40f', '#8e44ad', '#2980b9', '#27ae60', '#16a085']

def plt_data(data, step, title, c_id, savefig=True):
    x = [i for i in range(step)]
    plt.plot(x, data, color=color_list[c_id], label=title)
    if savefig:
        plt.savefig('savefig/' + title + '.png')


def get_log_data(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')
#         data_dict = {'pre_loss': [], 'g_loss': [], 'mana_loss': [], 'work_loss': [],
#                      'd_loss': [], 'train_acc': [], 'eval_acc': [], 'NLL_oracle': [],
#                      'NLL_gen': [], 'BLEU-3': [], 'BLEU' :[]}
        data_dict = {'BLEU' :[]}

        for line in all_lines:
            items = line.split()
            try:
                for key in data_dict.keys():
                    if key in items:
                        #data_dict[key].append(float(items[items.index(key) + 2][:-1]))
                        data_dict[key].append(float(items[items.index(key) + 5][:-1]))
            except:
                break
        #print(data_dict)

    return data_dict


if __name__ == '__main__':
    log_file_root = '../log/'
    # Custom your log files in lists, no more than len(color_list)
#     log_file_list = ['log_0604_2233', 'log_0605_0120', 'log_0531_1507']
#     legend_text = ['SeqGAN', 'LeakGAN', 'RelGAN']
#     log_file_list = ['log_1017_1209_SeqGAN','log_1016_2029_JSDGAN','log_1015_1125_DPGAN']
#     legend_text = ['SeqGAN', 'JSDGAN', 'DPGAN']
    log_file_list = ['log_1015_1125_DPGAN']
    legend_text = ['DPGAN']

    color_id = 0
    data_name = 'BLEU'
    row_name = 'BLEU'
    if_save = True
    # legend_text = log_file_list

    assert data_name in title_dict.keys(), 'Error data name'
    plt.clf()
    plt.title('Comparasion')
    all_data_list = []
    for idx, item in enumerate(log_file_list):
       
        log_file = item + '.txt'
        #log_file = log_file_root + item + '.txt'
        # save log file
        all_data = get_log_data(log_file) #entrega los valores del campo
        plt_data(all_data[title_dict[data_name]], len(all_data[title_dict[data_name]]),
                 legend_text[idx], color_id, if_save)
        color_id += 1

    plt.legend()
    plt.show()
    
    SeqGANy = [0.102, 0.105, 0.119, 0.11, 0.103, 0.11, 0.099, 0.097, 0.104, 0.105, 0.104, 0.102, 0.108, 0.116, 0.105, 0.105, 0.102, 0.11, 0.103, 0.102, 0.099, 0.103, 0.086, 0.091, 0.097, 0.097, 0.101, 0.098, 0.101, 0.088, 0.093, 0.087, 0.098, 0.094, 0.096, 0.086, 0.092, 0.101, 0.092, 0.106, 0.092, 0.1, 0.092, 0.094, 0.102, 0.094, 0.094, 0.091, 0.098, 0.093, 0.095, 0.097, 0.103, 0.099, 0.099, 0.098, 0.091, 0.088, 0.088, 0.095, 0.1, 0.093, 0.1, 0.094, 0.092, 0.096, 0.096, 0.101, 0.096, 0.098, 0.095, 0.099, 0.105, 0.097, 0.097, 0.1, 0.1, 0.104, 0.103, 0.098, 0.092, 0.101, 0.103, 0.106, 0.095, 0.1, 0.098, 0.111, 0.098, 0.097, 0.095, 0.101, 0.1, 0.091, 0.099, 0.097, 0.102, 0.103, 0.096, 0.099, 0.105]
    JSDGANy = [0.167, 0.215, 0.242, 0.225, 0.253, 0.24, 0.228, 0.231, 0.245, 0.267, 0.239, 0.228, 0.256, 0.249, 0.234, 0.247, 0.236, 0.243, 0.237, 0.249, 0.222, 0.23, 0.256, 0.243, 0.226, 0.235, 0.237, 0.226, 0.221, 0.23, 0.227, 0.237, 0.238, 0.236, 0.237, 0.226, 0.223, 0.229, 0.22, 0.241, 0.227, 0.231, 0.242, 0.241, 0.207, 0.216, 0.239, 0.225, 0.228, 0.22, 0.246, 0.238, 0.226, 0.217, 0.212, 0.222, 0.219, 0.23, 0.248, 0.246, 0.213, 0.223, 0.233, 0.219, 0.255, 0.211, 0.129, 0.236, 0.239, 0.251, 0.22, 0.233, 0.241, 0.218, 0.235, 0.235, 0.245, 0.227, 0.242, 0.237, 0.244, 0.238, 0.229, 0.224, 0.198, 0.215, 0.239, 0.248, 0.237, 0.219, 0.25, 0.235, 0.23, 0.252, 0.221, 0.224, 0.242, 0.246, 0.242, 0.222, 0.234]
    DPGANy = [0.121, 0.131, 0.138, 0.158, 0.189, 0.201, 0.213, 0.19, 0.213, 0.203, 0.231, 0.25, 0.232, 0.223, 0.231, 0.215, 0.221, 0.187, 0.197, 0.219, 0.2, 0.182, 0.18, 0.159, 0.164, 0.161, 0.148, 0.137, 0.14, 0.14, 0.132, 0.133, 0.116, 0.112, 0.112, 0.109, 0.092, 0.091, 0.091, 0.088, 0.084, 0.089, 0.082, 0.085, 0.076, 0.082, 0.084, 0.092, 0.092, 0.088, 0.092, 0.096, 0.097, 0.095, 0.096, 0.097, 0.094, 0.094, 0.097, 0.098, 0.1, 0.099, 0.102, 0.1, 0.095, 0.098, 0.096, 0.096, 0.096, 0.088, 0.094, 0.092, 0.088, 0.089, 0.085, 0.083, 0.078, 0.075, 0.08, 0.078, 0.072, 0.071, 0.065, 0.061, 0.06, 0.059, 0.059, 0.055, 0.058, 0.056, 0.056, 0.052, 0.053, 0.049, 0.054, 0.051, 0.05, 0.05, 0.043, 0.055, 0.053]
    
    def comparacion_grafico():
        plt.title('Comparación de puntajes BLEU-4')
        ejex = [i for i in range(len(all_data[title_dict[data_name]]))]
        plt.plot(ejex, SeqGANy, color='C0', label='SeqGAN')
        plt.plot(ejex, JSDGANy, color='C1', label='JSDGAN')
        plt.plot(ejex, DPGANy, color='C2', label='DPGAN')
        plt.legend(["SeqGAN", "JSDGAN", "DPGAN"])
        plt.xlabel("Época")
        plt.ylabel("Puntaje")
        plt.savefig('savefig/' + 'comparacion' + '.png')
     
    comparacion_grafico()