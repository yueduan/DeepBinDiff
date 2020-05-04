import sys
import os
import numpy as np
# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import utility
# import time

import preprocessing


if sys.platform != "win32":
    embedding_file = "./vec_all"
    func_embedding_file = "./func_vec_all"
    node2addr_file = "./data/DeepBD/nodeIndexToCode"
    func2addr_file = "./data/DeepBD/functionIndexToCode"
    bin_edgelist_file = "./data/DeepBD/edgelist"
    bin_features_file = "./data/DeepBD/features"
    func_features_file = "./data/DeepBD/func.features"
    ground_truth_file = "./data/DeepBD/addrMapping"
else:
    embedding_file = ".\\vec_all"
    func_embedding_file = ".\\func_vec_all"
    node2addr_file = ".\\data\\DeepBD\\nodeIndexToCode"
    func2addr_file = ".\\data\\DeepBD\\functionIndexToCode"
    bin_edgelist_file = ".\\data\\DeepBD\\edgelist"
    bin_features_file = ".\\data\\DeepBD\\features"
    func_features_file = ".\\data\\DeepBD\\func.features"
    ground_truth_file = ".\\data\\DeepBD\\addrMapping"

# whether use deepwalk to create embeddings for each function or not 
# Set to false as default, which can get better result for now.
EBD_CALL_GRAPH = False 


def pre_matching(bin1_name, bin2_name, toBeMergedBlocks={}):
    # if sys.platform != "win32":


    tadw_command = "python3 ./src/performTADW.py --method tadw --input " + bin_edgelist_file + " --graph-format edgelist --feature-file " + bin_features_file + " --output vec_all"
    os.system(tadw_command)
    
    ebd_dic, _ = utility.ebd_file_to_dic(embedding_file)

    node_in_bin1, _node_in_bin2 = utility.readNodeInfo(node2addr_file)
    
    
    bin1_mat = []
    bin2_mat = []
    node_map = {}
    for idx, line in ebd_dic.items():
        if idx < node_in_bin1:
            bin1_mat.append(line)
            node_map[str(idx)] = len(bin1_mat) - 1
        else:
            bin2_mat.append(line)
            node_map[str(idx)] = len(bin2_mat) - 1


    bin1_mat = np.array(bin1_mat)
    bin2_mat = np.array(bin2_mat)
    sim_result = utility.similarity_gpu(bin1_mat, bin2_mat)
    
    print("Perform matching...")
    matched_pairs, inserted, deleted = utility.matching(node_in_bin1, ebd_dic, sim_result, node_map, toBeMergedBlocks)

    print("matched pairs: ")
    print(matched_pairs)

    # print("Inserted blocks: ")
    # print(inserted)

    # print("Deleted blocks: ")
    # print(deleted)

   


# if __name__ == '__main__' :
#     # here is cross-platform configurations. 
#     # actually I can do this in more elegant way, but it is enough for testing.
    
#     # np.set_printoptions(threshold=np.inf, suppress=True)  # set numpy options
#     sys.exit(two_level_matching('yes_830_o1', 'yes_830_o3'))
