import numpy as np
import tensorflow as tf
import re
from scipy import spatial
import operator
import lapjv

import preprocessing


matching_threshold = 0.6

# read the embedding file
def ebd_file_to_dic(embedding_file):
    ebd_list = np.loadtxt(embedding_file)
    ebd_dic = {}
    feature_dim = len(ebd_list[0]) - 1
    for ebd in ebd_list:
        ebd_dic[int(ebd[0])] = ebd[1:]
    return ebd_dic, feature_dim


# similarity function cosine as default
def similarity(mat_a, mat_b):
    mul = np.dot(mat_a, mat_b.T)
    na = np.linalg.norm(mat_a, axis=1, keepdims=True)
    nb = np.linalg.norm(mat_b, axis=1, keepdims=True)
    result = mul / np.dot(na, nb.T)
    return result


# similarity function, GPU speed up
def similarity_gpu(mat_a, mat_b):
    funca = tf.placeholder(tf.float32, shape=mat_a.shape)
    funcb = tf.placeholder(tf.float32, shape=mat_b.shape)
    mul = tf.matmul(funca, funcb, transpose_b=True)
    na = tf.norm(funca, axis=1, keepdims=True)
    nb = tf.norm(funcb, axis=1, keepdims=True)
    result =  mul / tf.matmul(na, nb, transpose_b=True)
    with tf.Session() as sess:
        return sess.run(result, feed_dict={funca: mat_a, funcb: mat_b})


def merge(vec_list, method='sum'):
    def vec_max(vec_list):
        vec_list = np.array(vec_list)
        return vec_list.max(0)
    def vec_min(vec_list):
        vec_list = np.array(vec_list)
        return vec_list.min(0)
    def vec_sum(vec_list):
        vec_list = np.array(vec_list)
        return vec_list.sum(0) 
    func_switcher = {'max': vec_max, 'min': vec_min, 'sum': vec_sum}
    return func_switcher[method](vec_list)


def readFunc2AddrFile(bin1_name, bin2_name, func2addr_file, ebd_dic, bin1_functions, bin2_functions, bin1_nonimported_funcs, bin2_nonimported_funcs):
    with open(func2addr_file) as f:
        lines = f.readlines()
        idx_nonumported = 0
        for idx, line in enumerate(lines):
            function = {}
            if idx == 0:
                func_in_bin1, func_in_bin2 = int(line.split()[0]), int(line.split()[1])
                continue
            elif idx % 3 == 1:
                blocks = lines[idx+2].split()
                blocks = [b for b in blocks if int(b) in ebd_dic]
                if len(blocks):
                    function['blocks'] = blocks
                else:
                    function['blocks'] = None
                function['name'] = lines[idx+1].split()[1]
                function['addr'] = lines[idx+1].split()[2]
                function['import'] = not (lines[idx+1].split()[3] == bin1_name or lines[idx+1].split()[3] == bin2_name)
                binary = lines[idx+1].split()[0]
                no = int(lines[idx][:-2])
                if binary == 'Bin1':
                    bin1_functions[no] = function
                    if function['import'] != True and function['blocks'] != None:
                        bin1_nonimported_funcs[idx_nonumported] = function
                        idx_nonumported = idx_nonumported + 1
                elif binary == 'Bin2':
                    bin2_functions[no] = function
                    if function['import'] != True and function['blocks'] != None:
                        bin2_nonimported_funcs[idx_nonumported] = function
                        idx_nonumported = idx_nonumported + 1


def writeFuncFeatures(func_features, ebd_dic, feature_dim, bin1_nonimported_funcs, bin2_nonimported_funcs):
    print("Writing func features..")
    with open(func_features, 'w') as f:
        for function in bin1_nonimported_funcs.items():
            func_feature = []
            if function[1]['blocks'] is not None:
                for block in function[1]['blocks']:
                    if int(block) in ebd_dic:
                        func_feature.append(ebd_dic[int(block)]) 
                    else:
                        print('remove', block)
                        # bin1_functions[function[0]]['blocks'].remove(block)
            # 'activation function'
            func_feature = merge(func_feature)
            f.write(str(function[0]) + ' ')
            for item in func_feature:
                f.write(str(item) + ' ')
            f.write('\n')
        for function in bin2_nonimported_funcs.items():
            func_feature = []
            if function[1]['blocks'] is not None:
                for block in function[1]['blocks']:
                    if int(block) in ebd_dic:
                        func_feature.append(ebd_dic[int(block)])
                    else:
                        print('remove', block)
                        # bin2_functions[function[0]]['blocks'].remove(block)
            # 'activation function'
            func_feature = merge(func_feature)
            f.write(str(function[0]) + ' ')
            for item in func_feature:
                f.write(str(item) + ' ')
            f.write('\n')


def readNodeInfo(node2addr_file):
    bb_list = []
    bb_op_list = []
    # read the instructions and address
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
                    i = int(ins.split()[-1], 16)# - 4194304
                    bb_addr_vec.append(hex(i))


                bb_list.append(bb_addr_vec)
                bb_op_list.append(bb_op_vec)
    
    return node_in_bin1, node_in_bin2


# This function matches functions and further matches basic blocks inside function pairs
def matching(node_in_bin1, ebd_dic, sim_result, node_map, toBeMergedBlocks):
    # matchBBs(bin1_nonimported_funcs, bin2_nonimported_funcs, matchFuncs(bin1_nonimported_funcs, bin2_nonimported_funcs, func_ebd_dic, sim_result, func_map), matches_list, sim_result, node_map)

    # get k hop neighbors and perform greedy matching algorithm
    k_hop_neighbors = get_k_neighbors()
    matched_pairs, matched_blocks = k_hop_greedy_matching(k_hop_neighbors, sim_result, node_map, toBeMergedBlocks)

    #perform linear assignment for the unmatched blocks
    non_matched_bin1 = []
    non_matched_bin2 = []

    # mapping between the node id and node index in non_matched_binx
    non_matched_to_global_dict1 = {}
    non_matched_to_global_dict2 = {}

    index1 = 0
    index2 = 0
    for bid in ebd_dic:
        if bid not in matched_blocks:
            if bid < node_in_bin1:
                non_matched_bin1.append(bid)
                non_matched_to_global_dict1[index1] = bid
                index1 += 1

            else:
                non_matched_bin2.append(bid)
                non_matched_to_global_dict2[index2] = bid
                index2 += 1

    cost_array = []
    maxNumber = max(len(non_matched_bin1), len(non_matched_bin2))
    for _idx in range(maxNumber):
        per_node_list = [0] * maxNumber
        cost_array.append(per_node_list)

    for idx_1, bb1 in enumerate(non_matched_bin1):
        for idx_2, bb2 in enumerate(non_matched_bin2):
            sim = sim_result[node_map[str(bb1)], node_map[str(bb2)]]
            cost_array[idx_1][idx_2] = 1 - sim

    for idx_1, bb1 in enumerate(non_matched_bin1):
        # print("adding {} for {} in dict1".format(idx_1, int(bb1)))
        non_matched_to_global_dict1[idx_1] = int(bb1)
    for idx_2, bb2 in enumerate(non_matched_bin2):
        # print("adding {} for {} in dict2".format(idx_2, int(bb2)))
        non_matched_to_global_dict2[idx_2] = int(bb2)

    mapping = linearAssignment(cost_array)
    bb_matching_pair = []
    # check if the it is inserted or deleted
    # only put the matched pairs into 'bb_matching_pair'
    for k in range(maxNumber):
        match_bb1 = -1
        match_bb2 = -1

        if k < len(non_matched_bin1):
            match_bb1 = k
        if mapping[k] < len(non_matched_bin2):
            match_bb2 = mapping[k]
    
        # if match_bb1 == -1:
        #     print("Node: {} is inserted".format(nonimported_to_global_dict_2[match_bb2]))
        # elif match_bb2 == -1:
        #     print("Node: {} is deleted".format(nonimported_to_global_dict_1[match_bb1]))
        # else:
        if match_bb1 != -1 and match_bb2 != -1:
            bb_matching_pair.append([match_bb1, match_bb2])

    inserted = []
    deleted = []

    return matched_pairs, inserted, deleted






# get the k-hop neighbors in the CFG for every block (bid)
def get_k_neighbors(k=4):
    # k_hop_preds_dict = {}
    # k_hop_succs_dict = {}
    k_hop_neighbors = {}
    
    for block_id in preprocessing.per_block_neighbors_bids:
        curr_bids = []
        # print("block_id", block_id)
        # get k hop predecessors
        k_preds = []
        curr_bids.append(block_id)

        for _ in range(0, k):
            next_level_bids = curr_bids.copy()
            curr_bids.clear()
            # print("next_level_bids:", next_level_bids)
            for nl_bid in next_level_bids:
                preds = preprocessing.per_block_neighbors_bids[nl_bid][0]

                for pred in preds:
                    if pred not in k_preds:
                        k_preds.append(pred)
                    if pred not in curr_bids:
                        curr_bids.append(pred)
            # print("curr_bids:", curr_bids)

        
        # print("preds:", k_hop_preds, "\n")

        # get k hop successors
        k_succs = []
        curr_bids.clear()
        curr_bids.append(block_id)
        # print('curr_bids initial:', curr_bids)
        for _ in range(0, k):
            next_level_bids = curr_bids.copy()
            curr_bids.clear()
            # print("next_level_bids:", next_level_bids)
            for nl_bid in next_level_bids:
                succs = preprocessing.per_block_neighbors_bids[nl_bid][1]
                for succ in succs:
                    if succ not in k_succs:
                        k_succs.append(succ)
                    if succ not in curr_bids:
                        curr_bids.append(succ)

            # print("curr_bids:", curr_bids)

        # print("succs:", k_hop_succs, "\n")

        # k_hop_preds_dict[block_id] = k_preds
        # k_hop_succs_dict[block_id] = k_succs
        k_hop_neighbors[block_id] = k_preds + k_succs

    return k_hop_neighbors#k_hop_preds_dict, k_hop_succs_dict


def find_max_unmatched_pair(sims, matched_blocks):
    for key, value in sorted(sims.items(), key = lambda kv:(kv[1], kv[0]), reverse=True):
        if key[0] in matched_blocks or key[1] in matched_blocks:
            continue
        if value < matching_threshold:
            return []
        return list(key)
    return []

# This function matches the blocks in a k-hop greedy way. 
# Consider the definite matches (using API call and long string) as initial set
# we then start to match their surrouding k-hop nodes using embeddings until all nodes are matched
def k_hop_greedy_matching(k_hop_neighbors, sim_result, node_map, toBeMergedBlocks):    
    # we start from toBeMergedBlocks since they are definite matches
    matched_pairs = [] # this list stores all the matched block pairs
    matched_blocks = [] # this list stores all the blocks that are already matched

    for bid1 in toBeMergedBlocks:
        pair = [bid1, toBeMergedBlocks[bid1]]
        matched_pairs.append(pair)
        matched_blocks.append(bid1)
        matched_blocks.append(toBeMergedBlocks[bid1])


    curr_pairs = matched_pairs.copy()

    while len(curr_pairs) != 0:
        curr_pair = curr_pairs.pop(0)

        # preds1_ids = k_hop_preds_dict[curr_pair[0]]
        # preds2_ids = k_hop_preds_dict[curr_pair[1]]
        
        # succs1_ids = k_hop_succs_dict[curr_pair[0]]
        # succs2_ids = k_hop_succs_dict[curr_pair[1]]

        neighbor1_ids = k_hop_neighbors[curr_pair[0]]
        neighbor2_ids = k_hop_neighbors[curr_pair[1]]


        sims = {}
        for bid1 in neighbor1_ids:
            for bid2 in neighbor2_ids:
                bid_pair = (bid1, bid2)
                if bid1 in preprocessing.non_code_block_ids or bid2 in preprocessing.non_code_block_ids:
                    sims[bid_pair] = 0
                else:
                    sims[bid_pair] = sim_result[node_map[str(bid1)], node_map[str(bid2)]]
        
        # for bid1 in succs1_ids:
        #     for bid2 in succs2_ids:
        #         bid_pair = (bid1, bid2)
        #         if bid1 in preprocessing.non_code_block_ids or bid2 in preprocessing.non_code_block_ids:
        #             sims[bid_pair] = 0
        #         else:
        #             sims[bid_pair] = sim_result[node_map[str(bid1)], node_map[str(bid2)]]



        maxpair = find_max_unmatched_pair(sims, matched_blocks)

        if not maxpair:
            continue 
        
        matched_pairs.append(maxpair)
        matched_blocks.append(maxpair[0])
        matched_blocks.append(maxpair[1])
        curr_pairs.append(maxpair)
        curr_pairs.append(curr_pair)

    return matched_pairs, matched_blocks


# Greedy version
# def matchFuncs(bin1_nonimported_funcs, bin2_nonimported_funcs, func_ebd_dic, sim_result, func_map):
#     function_matching_pair = []
#     for func1 in bin1_nonimported_funcs.items():
#         func_result_list = {}
#         for func2 in bin2_nonimported_funcs.items():
#             if (func1[1]['blocks'] != None) and (func2[1]['blocks'] != None) and (func1[1]['import'] != True) and (func2[1]['import'] != True):
#                 sim = sim_result[func_map[str(func1[0])], func_map[str(func2[0])]]
#                 func_result_list[str(func1[0]) + ' ' + str(func2[0])] =  sim
#                 if func1[1]['name'] == 'put_indicator':
#                     if func2[1]['name'] == 'put_indicator':
#                         print("xjtu")
#                     print("func in bin1: {}\nfunc in bin2: {}\nhas similarity: {}\n\n".format(func1, func2, sim))
#         sorted_list = sorted(func_result_list.items(), key=lambda item:item[1])
#         if len(sorted_list):
#             match_func1, match_func2 = sorted_list[-1][0].split()
#             # print("match: {} and {}".format(bin1_nonimported_funcs[int(match_func1)], bin2_nonimported_funcs[int(match_func2)]))
#             function_matching_pair.append([int(match_func1), int(match_func2)])
#     return function_matching_pair


# Linear assignment version
def matchFuncs(bin1_nonimported_funcs, bin2_nonimported_funcs, func_ebd_dic, sim_result, func_map):
    cost_array = []
    print("bin1 nonimported function len: ", len(bin1_nonimported_funcs))
    print("bin2 nonimported function len: ", len(bin2_nonimported_funcs))
    maxNumber = max(len(bin1_nonimported_funcs), len(bin2_nonimported_funcs))
    print(maxNumber)
    for _idx in range(maxNumber):
        per_func_list = [0] * maxNumber
        cost_array.append(per_func_list)
    
    for func1 in bin1_nonimported_funcs.items():
        # use function embeddings to calculate similarities
        func_result_list = {}
        func_sim_list = {}
        for func2 in bin2_nonimported_funcs.items():
            # func1[0], func2[0] are the indices
            if (func1[1]['blocks'] != None) and (func2[1]['blocks'] != None) and (func1[1]['import'] != True) and (func2[1]['import'] != True):
                sim = sim_result[func_map[str(func1[0])], func_map[str(func2[0])]]
                func_result_list[str(func1[0]) + ' ' + str(func2[0])] =  sim
                # if func1[1]['name'] == 'version_etc_va':
                func_sim_list[func2[1]['name']] = sim
                    # print("func in bin1: {}\nfunc in bin2: {}\nhas similarity: {}\n\n".format(func1, func2, sim))
                cost_array[func1[0]][func2[0] - len(bin1_nonimported_funcs)] = 1 - sim
        sorted_x = sorted(func_sim_list.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
        # if func1[1]['name'] == 'version_etc_va':
        # print("sort {}\n\n\n".format(func1[1]['name']))
        # for f2n in sorted_x:
        #     for func2 in bin2_nonimported_funcs.items():
        #         if func2[1]['name'] == f2n[0]:
        #             print("func in bin1: {}\nfunc in bin2: {}\nsimilarity: {}\n\n".format(func1, func2, f2n[1]))

        
    mapping = linearAssignment(cost_array)
    function_matching_pair = []
    maxNumber = max(len(bin1_nonimported_funcs), len(bin2_nonimported_funcs))
    # check if the function is inserted or deleted
    # only put the matched pairs into 'function_matching_pair'
    for k in range(maxNumber):
        match_func1 = -1
        match_func2 = -1

        if(k < len(bin1_nonimported_funcs)):
            match_func1 = k
        if(mapping[k] < len(bin2_nonimported_funcs)):
            match_func2 = mapping[k] + len(bin1_nonimported_funcs)
        
        if match_func1 == -1:
            print("Func: {} is inserted".format(bin2_nonimported_funcs[match_func2]))
        elif match_func2 == -1:
            print("Func: {} is deleted".format(bin1_nonimported_funcs[match_func1]))
        else:
        #if match_func1 != -1 and match_func2 != -1:
            function_matching_pair.append([match_func1, match_func2])
    return function_matching_pair


# work on the matched functions for the second level (basic block level) matching
def matchBBs(bin1_nonimported_funcs, bin2_nonimported_funcs, function_matching_pair, matches_list, sim_result, node_map):
    print("total matched blocks: {}".format(len(matches_list)))
    matched_sims = []
    with open("2lvmatchResult.txt", 'w') as result_file:
        for pair in function_matching_pair:
            match_func1 = pair[0]
            match_func2 = pair[1]
            # print("match func # {} from bin1: {} \n\t and func # {} from bin 2: {}\n".format(match_func1, bin1_nonimported_funcs[match_func1], match_func2, bin2_nonimported_funcs[match_func2]))

            if bin1_nonimported_funcs[match_func1]['name'] != bin2_nonimported_funcs[match_func2]['name']:
                print("matched:\n\tfunc from bin1: {} \n\tfunc from bin2: {}\n\n".format(bin1_nonimported_funcs[match_func1], bin2_nonimported_funcs[match_func2]))

            # these two structures store the mapping between indices of nonimported and real
            nonimported_to_global_dict_1 = {}
            nonimported_to_global_dict_2 = {}

            cost_array = []
            maxNumber = max(len(bin1_nonimported_funcs[match_func1]['blocks']), len(bin2_nonimported_funcs[match_func2]['blocks']))
            for _idx in range(maxNumber):
                per_func_list = [0] * maxNumber
                cost_array.append(per_func_list)

            for idx_1, bb1 in enumerate(bin1_nonimported_funcs[match_func1]['blocks']):
                # bb_result_list = {}
                for idx_2, bb2 in enumerate(bin2_nonimported_funcs[match_func2]['blocks']):
                    sim = sim_result[node_map[bb1], node_map[bb2]]
                    #bb_result_list[bb1 + ' ' + bb2] =  sim
                    cost_array[idx_1][idx_2] = 1 - sim

            for idx_1, bb1 in enumerate(bin1_nonimported_funcs[match_func1]['blocks']):
                # print("adding {} for {} in dict1".format(idx_1, int(bb1)))
                nonimported_to_global_dict_1[idx_1] = int(bb1)
            for idx_2, bb2 in enumerate(bin2_nonimported_funcs[match_func2]['blocks']):
                # print("adding {} for {} in dict2".format(idx_2, int(bb2)))
                nonimported_to_global_dict_2[idx_2] = int(bb2)

            mapping = linearAssignment(cost_array)
            bb_matching_pair = []
            # check if the function is inserted or deleted
            # only put the matched pairs into 'bb_matching_pair'
            for k in range(maxNumber):
                match_bb1 = -1
                match_bb2 = -1

                if(k < len(bin1_nonimported_funcs[match_func1]['blocks'])):
                    match_bb1 = k
                if(mapping[k] < len(bin2_nonimported_funcs[match_func2]['blocks'])):
                    match_bb2 = mapping[k]
            
                # if match_bb1 == -1:
                #     print("Node: {} is inserted".format(nonimported_to_global_dict_2[match_bb2]))
                # elif match_bb2 == -1:
                #     print("Node: {} is deleted".format(nonimported_to_global_dict_1[match_bb1]))
                # else:
                if match_bb1 != -1 and match_bb2 != -1:
                    bb_matching_pair.append([match_bb1, match_bb2])

            sim = 0
            for pair in bb_matching_pair:
                pair = [nonimported_to_global_dict_1[pair[0]], nonimported_to_global_dict_2[pair[1]]]
                
                matched_sims.append(sim_result[node_map[str(pair[0])], node_map[str(pair[1])]])
    
                # pair = list(map(int, pair))
                # if pos >= length - 5:
                    # print("high rank:", int(pair[0]), "==", int(pair[1]),"score:", tpl[1], "rank:", length - pos)
                if pair in matches_list:
                    # print("----------------------------------------")
                    # print("ground truth:", int(pair[0]), "==", int(pair[1]), "score:", tpl[1], "rank:", length - pos)
                    result_file.write("successful matching ground truth: {} == {}\n".format(pair[0], pair[1]))
                    # print("----------------------------------------")
    
    print("matched average sim: ", sum(matched_sims)/len(matched_sims))
    matched_sims.sort()


# perform linear assignment for the given cost array
def linearAssignment(cost_array):
    cost_matrix = np.array([np.array(xi) for xi in cost_array], dtype=np.float32)
    print(cost_matrix)
    print(cost_matrix.shape)
    row_ind, _col_ind, _ = lapjv.lapjv(cost_matrix)
    return row_ind
