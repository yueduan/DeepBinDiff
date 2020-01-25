import sys
import os
import numpy as np
from scipy import spatial
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial
import re
import multiprocessing as mp
import lapjv
import time

# similarity function cosine as default
def similarity(vector1, vector2, dtype="cosine"):
    if dtype == "cosine":
        return 1 - spatial.distance.cosine(vector1, vector2)
    elif dtype == "Euclidean":
        return np.sqrt(np.sum(np.square(vector1-vector2)))


# read the embedding file
def ebd_file_to_dic(inputfile):
    ebd_list = np.loadtxt(inputfile)
    ebd_dic = {}
    feature_dim = len(ebd_list[0]) - 1   
    for ebd in ebd_list:
        ebd_dic[int(ebd[0])] = ebd[1:]
    return ebd_dic, feature_dim


def match(node_in_bin1, node_in_bin2, matches_list, ebd_dic):#, idx_x):
    # result_dict = {}
    # for idx_y in range(node_in_bin1, node_in_bin1 + node_in_bin2):  
    #     if idx_x in ebd_dic and idx_y in ebd_dic:
    #         sim = similarity(ebd_dic[idx_x], ebd_dic[idx_y])
    #         result_dict[str(idx_x) + ' ' + str(idx_y)] =  sim
    # sorted_list = sorted(result_dict.items(), key=lambda item:item[1])
    # length = len(sorted_list)
    # # print(sorted_list)
    # for pos, tpl in enumerate(sorted_list):
    #     pair = tpl[0].split()
    #     pair = list(map(int, pair))
    #     # if pos >= length - 5:
    #         # print("high rank:", int(pair[0]), "==", int(pair[1]), "score:", tpl[1], "rank:", length - pos)
    #     if pair in matches_list:
    #         # print("----------------------------------------")
    #         with open("matchResult.txt", 'a') as f:
    #             # print("ground truth:", int(pair[0]), "==", int(pair[1]), "score:", tpl[1], "rank:", length - pos)
    #             f.write("ground truth: {} == {}, score: {}, rank: {}\n".format(int(pair[0]),int(pair[1]), tpl[1], (length - pos)))
    #         # print("----------------------------------------")
    cost_array = []
    maxNumber = max(node_in_bin1, node_in_bin2)
    for idx_x in range(maxNumber):
        per_node_list = []
        for idx_y in range(maxNumber):
            per_node_list.append(0)
        cost_array.append(per_node_list)
    f = open("matchResult.txt", "w")
    for idx_x in range(node_in_bin1):
        result_dict = {}
        for idx_y in range(node_in_bin1, node_in_bin1 + node_in_bin2):  
            if idx_x in ebd_dic and idx_y in ebd_dic:
                sim = similarity(ebd_dic[idx_x], ebd_dic[idx_y])
            else:
                sim = 0
            result_dict[str(idx_x) + ' ' + str(idx_y)] = sim
            cost_array[idx_x][idx_y - node_in_bin1] = 1 - sim


    cost_matrix = np.array([np.array(xi) for xi in cost_array], dtype=np.float32)


    t1 = time.time()
    row_ind, _col_ind, _ = lapjv.lapjv(cost_matrix)
    t2 = time.time()
    print("linear assignment time: ", t2-t1)

    # [bb1, bb2]
    match_pair = []
    for k in range(maxNumber):
        oldIdx = -1
        newIdx = -1
        if(k < node_in_bin1):
            oldIdx = k
        if(row_ind[k] < node_in_bin2):
            newIdx = row_ind[k] + node_in_bin1
        if oldIdx == -1:
            print("Node: {} is inserted".format(newIdx))
        elif newIdx == -1:
            print("Node: {} is deleted".format(oldIdx))
        else:
            match_pair.append([oldIdx, newIdx])
            print("Node: {} is matched to Node:{}".format(oldIdx, newIdx))

    for pair in match_pair:
        pair = list(map(int, pair))
        if pair in matches_list:
            f.write("matched between Node {} and Node {}".format(pair[0], pair[1]))

        # sorted_list = sorted(result_dict.items(), key=lambda item:item[1])
        # length = len(sorted_list)
        # # print(sorted_list)
        # for pos, tpl in enumerate(sorted_list):
        #     pair = tpl[0].split()
        #     pair = list(map(int, pair))
        #     # if pos >= length - 5:
        #         # print("high rank:", int(pair[0]), "==", int(pair[1]), "score:", tpl[1], "rank:", length - pos)
        #     if pair in matches_list:
        #         # print("----------------------------------------")
        #         # print("ground truth:", int(pair[0]), "==", int(pair[1]), "score:", tpl[1], "rank:", length - pos)
        #         f.write("ground truth: {} == {}, score: {}, rank: {}\n".format(int(pair[0]),int(pair[1]), tpl[1], (length - pos)))
        #         # print("----------------------------------------")
    f.close()


def main(argv=None):
    if sys.platform != "win32":
        os.system("python3 ./src/main.py --method tadw --input data/DeepBD/cal_edgelist.txt --graph-format edgelist --feature-file data/DeepBD/cal.features --output vec_all.txt")
    else:
        os.system("python .\src\main.py --method tadw --input data\DeepBD\cal_edgelist.txt --graph-format edgelist --feature-file data\DeepBD\cal.features --output vec_all.txt")

    ebd_dic = ebd_file_to_dic(embedding_file)[0]

    print("TADW done")


    bb_list = []
    bb_op_list = []

    # first read the instructions and address
    with open(node2addr_file) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx == 0:
                node_in_bin1, node_in_bin2 = int(line.split()[0]), int(line.split()[1])
            elif idx % 3 == 2:
                bb_addr_vec = []
                bb_op_vec = []
                line_vec = re.findall('<(.*?)>', line)
                bb_op_vec.append(int(lines[idx-1][:-2]))
                bb_addr_vec.append(int(lines[idx-1][:-2]))
                for ins in line_vec:
                    bb_op_vec.append(ins.split()[1])
                    i = int(ins.split()[-1], 16) - 4194304
                    bb_addr_vec.append(hex(i))
                bb_list.append(bb_addr_vec)
                bb_op_list.append(bb_op_vec)

    matches_list = []
    print("Building ground truth...")
    # then read the ground truth
    with open(ground_truth) as f:
        for line in f.readlines():
            pair = re.findall('\[(.*?)\]', line)
            #print("pair: {} {}".format(pair[0], pair[1]))
            assert len(pair) == 2
            original_addr_list = pair[0].split(', ')
            mod_addr_list = pair[1].split(', ')
            ori_bb_list = []
            mod_bb_list = []
            for item in original_addr_list:
                addr = hex(int(item))
                # print(addr)
                for bb in bb_list:
                    if (str(addr) in bb[1:]) and (bb[0] < node_in_bin1): 
                        ori_bb_list.append(bb[0])

            for item in mod_addr_list:
                addr = hex(int(item))
                # print(addr)
                for bb in bb_list:
                    if (str(addr) in bb[1:]) and (bb[0] >= node_in_bin1):
                        mod_bb_list.append(bb[0])
            # print(ori_bb_list)
            # print(mod_bb_list)
            # print("===========================================")
            if len(mod_bb_list) == 1 and len(ori_bb_list) ==1: 
                for bb1 in ori_bb_list:
                    for bb2 in mod_bb_list:
                        if ([bb1, bb2] not in matches_list):
                            matches_list.append([bb1, bb2])   
            elif len(mod_bb_list) == 0 or len(ori_bb_list) ==0:   
                continue     
            else:
                for bb1 in ori_bb_list:
                    for bb2 in mod_bb_list:
                        bb1_op = [i for i in bb_op_list if bb1 == i[0]]
                        bb2_op = [i for i in bb_op_list if bb2 == i[0]]
                        if ([bb1, bb2] not in matches_list) and bb1_op[0][1:] == bb2_op[0][1:]:
                            matches_list.append([bb1, bb2])

    print("Generating match results...")
    if os.path.isfile('matchResult.txt'):
        os.remove('matchResult.txt')
    match(node_in_bin1,node_in_bin2,matches_list,ebd_dic)
    # lock = mp.Lock()
    # pool = mp.Pool(processes=4)
    # job = partial(match, node_in_bin1, node_in_bin2, matches_list, ebd_dic)
    # res = pool.map(job, range(node_in_bin1))
    # pool.close()
    # pool.join()


if __name__ == '__main__' :
    # here is cross-platform configurations. 
    # actually I can do this in more elegant way, but it is enough for testing.
    if sys.platform != "win32":
        embedding_file = "./vec_all.txt"
        node2addr_file = "./data/DeepBD/nodeIndexToCode"
        bin_edgelist = "./data/DeepBD/cal_edgelist.txt"
        bin_features = "./data/DeepBD/cal.features"
        ground_truth = "./data/DeepBD/addrMapping"

    else:
        embedding_file = ".\\vec_all.txt"
        node2addr_file = ".\\data\\DeepBD\\nodeIndexToCode"
        bin_edgelist = ".\\data\\DeepBD\\cal_edgelist.txt"
        bin_features = ".\\data\\DeepBD\\cal.features"
        ground_truth = ".\\data\\DeepBD\\addrMapping"
    np.set_printoptions(threshold=np.inf, suppress=True)  # set numpy options
    sys.exit(main(sys.argv[1:]))