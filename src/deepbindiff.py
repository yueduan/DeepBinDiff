import os
import collections
import ntpath
import math

from shutil import copyfile
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import matching_driver
import featureGen
import preprocessing
from deepwalk import deepwalk


import tensorflow as tf
import numpy as np


# this list contains all the indices of the opcode in opcode_list
opcode_idx_list = []


# boundaryIdx = -1

# blockIdxToTokens: blockIdxToTokens[block index] = token list
# return dictionary: index to token, reversed_dictionary: token to index
def vocBuild(blockIdxToTokens):
    global opcode_idx_list
    vocabulary = []
    reversed_dictionary = dict()
    count = [['UNK'], -1]
    index = 0
    for idx in blockIdxToTokens:
        for token in blockIdxToTokens[idx]:
            vocabulary.append(token)
            if token not in reversed_dictionary:
                reversed_dictionary[token] = index
                if token in preprocessing.opcode_list and index not in opcode_idx_list:
                    opcode_idx_list.append(index)
                    # print("token:", token, " has idx: ", str(index))
                index = index + 1
                
    dictionary = dict(zip(reversed_dictionary.values(), reversed_dictionary.keys()))
    count.extend(collections.Counter(vocabulary).most_common(1000 - 1))
    print('20 most common tokens: ', count[:20])

    del vocabulary

    return dictionary, reversed_dictionary


# generate article for word2vec. put all random walks together into one article.
# we put a tag between blocks
def articlesGen(walks, blockIdxToTokens, reversed_dictionary):
    # stores all the articles, each article itself is a list
    article = []

    # stores all the block boundary indice. blockBoundaryIndices[i] is a list to store indices for articles[i].
    # each item stores the index for the last token in the block
    blockBoundaryIdx = []
    
    for walk in walks:
        # one random walk is served as one article
        for idx in walk:
            if idx in blockIdxToTokens:
                tokens = blockIdxToTokens[idx]
                for token in tokens:
                    article.append(reversed_dictionary[token])
            blockBoundaryIdx.append(len(article) - 1)
            # aritcle.append(boundaryIdx)
        
    insnStartingIndices = []
    indexToCurrentInsnsStart = {}
    # blockEnd + 1 so that we can traverse to blockEnd
    # go through the current block to retrive instruction starting indices
    for i in range(0, len(article)): 
        if article[i] in opcode_idx_list:
            insnStartingIndices.append(i)
        indexToCurrentInsnsStart[i] = len(insnStartingIndices) - 1

    
    # for counter, value in enumerate(insnStartingIndices):
    #     if data_index == value:
    #         currentInsnStart = counter
    #         break
    #     elif data_index < value:
    #         currentInsnStart = counter - 1
    #         break

    return article, blockBoundaryIdx, insnStartingIndices, indexToCurrentInsnsStart


# adopt TF-IDF method during block embedding calculation
def cal_block_embeddings(blockIdxToTokens, blockIdxToOpcodeNum, blockIdxToOpcodeCounts, insToBlockCounts, tokenEmbeddings, reversed_dictionary):
    block_embeddings = {}
    totalBlockNum = len(blockIdxToOpcodeCounts)


    for bid in blockIdxToTokens:
        tokenlist = blockIdxToTokens[bid]
        opcodeCounts = blockIdxToOpcodeCounts[bid]
        opcodeNum = blockIdxToOpcodeNum[bid]

        opcodeEmbeddings = []
        operandEmbeddings = []

        if len(tokenlist) != 0:
            for token in tokenlist:
                tokenid = reversed_dictionary[token]

                tokenEmbedding = tokenEmbeddings[tokenid]

                if tokenid in opcode_idx_list:
                    # here we multiple the embedding with its TF-IDF weight if the token is an opcode
                    tf_weight = opcodeCounts[token] / opcodeNum
                    x = totalBlockNum / insToBlockCounts[token]
                    idf_weight = math.log(x)
                    tf_idf_weight = tf_weight * idf_weight
                    # print("tf-idf: ", token, opcodeCounts[token], opcodeNum, totalBlockNum, insToBlockCounts[token], tf_weight, idf_weight)

                    opcodeEmbeddings.append(tokenEmbedding * tf_idf_weight)
                else:
                    operandEmbeddings.append(tokenEmbedding)

            opcodeEmbeddings = np.array(opcodeEmbeddings)
            operandEmbeddings = np.array(operandEmbeddings)

            opcode_embed = opcodeEmbeddings.sum(0)
            operand_embed = operandEmbeddings.sum(0)
        # set feature vector for null block node to be zeros
        else:
            embedding_size = 64
            opcode_embed = np.zeros(embedding_size)
            operand_embed = np.zeros(embedding_size)

        # if no operand, give zeros
        if operand_embed.size == 1:
            operand_embed = np.zeros(len(opcode_embed))
        

        block_embed = np.concatenate((opcode_embed, operand_embed), axis=0)
        block_embeddings[bid] = block_embed
        # print("bid", bid, "block embedding:", block_embed)


    return block_embeddings



def feature_vec_file_gen(feature_file, block_embeddings):
    with open(feature_file,'w') as feaVecFile:

        for counter in block_embeddings:
            value = block_embeddings[counter]
            # index as the first element and then output all the features
            feaVecFile.write(str(counter) + " ")
            for k in range(len(value)):
                feaVecFile.write(str(value[k]) + " ")
            feaVecFile.write("\n")


def copyEverythingOver(src_dir, dst_dir):
    # ground_truth = 'addrMapping'
    node_features = 'features'
    cfg_edgelist = 'edgelist_merged_tadw'
    #func_edgelist = 'func_edgelist'
    #functionInfo = 'functionIndexToCode'
    nodeInfo = 'nodeIndexToCode'

    #copyfile('/home/yueduan/yueduan/groundTruthCollection/output/' + ground_truth, dst_dir + ground_truth)
    # copyfile(src_dir + ground_truth, dst_dir + ground_truth)
    copyfile(src_dir + node_features, dst_dir + node_features)
    copyfile(src_dir + cfg_edgelist, dst_dir + 'edgelist')
    #copyfile(src_dir + func_edgelist, dst_dir + func_edgelist)
    #copyfile(src_dir + functionInfo, dst_dir + functionInfo)
    copyfile(src_dir + nodeInfo, dst_dir + nodeInfo)

    #Yue: use feature as embedding
    # copyfile(src_dir + node_features, 'vec_all')

def main():
    # example:
    # python3 src/deepbindiff.py --input1 input/ls_6.4 --input2 input/ls_8.30 --outputDir output/

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--input1', required=True, help='Input bin file 1')
    
    parser.add_argument('--input2', required=True, help='Input bin file 2')

    parser.add_argument('--outputDir', required=True, help='Specify the output directory') 
    args = parser.parse_args()
    filepath1 = args.input1
    filepath2 = args.input2
    outputDir = args.outputDir


    if outputDir.endswith('/') is False:
        outputDir = outputDir + '/'


    EDGELIST_FILE = outputDir + "edgelist"


    # step 1: perform preprocessing for the two binaries
    blockIdxToTokens, blockIdxToOpcodeNum, blockIdxToOpcodeCounts, insToBlockCounts, _, _, bin1_name, bin2_name, toBeMergedBlocks = preprocessing.preprocessing(filepath1, filepath2, outputDir)

    #step 2: vocabulary buildup
    dictionary, reversed_dictionary = vocBuild(blockIdxToTokens)

    # step 3: generate random walks, each walk contains certain blocks
    walks = deepwalk.randomWalksGen(EDGELIST_FILE, blockIdxToTokens)

    # step 4: generate articles based on random walks
    article, blockBoundaryIndex, insnStartingIndices, indexToCurrentInsnsStart = articlesGen(walks, blockIdxToTokens, reversed_dictionary)

    # step 5: token embedding generation
    tokenEmbeddings = featureGen.tokenEmbeddingGeneration(article, blockBoundaryIndex, insnStartingIndices, indexToCurrentInsnsStart, dictionary, reversed_dictionary, opcode_idx_list)

    # step 6: calculate feature vector for blocks
    block_embeddings = cal_block_embeddings(blockIdxToTokens, blockIdxToOpcodeNum, blockIdxToOpcodeCounts, insToBlockCounts, tokenEmbeddings, reversed_dictionary)
    feature_vec_file_gen(outputDir + 'features', block_embeddings) 

    copyEverythingOver(outputDir, 'data/DeepBD/')

    # step 7: TADW for block embedding generation & block matching
    matching_driver.pre_matching(bin1_name, bin2_name, toBeMergedBlocks)


if __name__ == "__main__":
    main()