import angr
import os
import ntpath
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# this list contains all the opcode in the two binaries
opcode_list = []

# this dictionary stores the predecessors and successors of nodes
# per_block_neighbors_bids[block_id] = [[predecessors],[successors]]
per_block_neighbors_bids = {}

# blocks that have no code
non_code_block_ids = []


# register list
register_list_8_byte = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']

register_list_4_byte = ['eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp', 'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d']

register_list_2_byte = ['ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp', 'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w']

register_list_1_byte = ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl', 'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def angrGraphGen(filepath1, filepath2):
    prog1 = angr.Project(filepath1, load_options={'auto_load_libs': False})
    prog2 = angr.Project(filepath2, load_options={'auto_load_libs': False})

    print("Analyzing the binaries to generate CFGs...")
    cfg1 = prog1.analyses.CFGFast()
    cg1 = cfg1.functions.callgraph
    print("First binary done")
    cfg2 = prog2.analyses.CFGFast()
    cg2 = cfg2.functions.callgraph
    print("CFGs Generated!")

    nodelist1 = list(cfg1.graph.nodes)
    edgelist1 = list(cfg1.graph.edges)

    nodelist2 = list(cfg2.graph.nodes)
    edgelist2 = list(cfg2.graph.edges)
    return cfg1, cg1, nodelist1, edgelist1, cfg2, cg2, nodelist2, edgelist2




def nodeDicGen(nodelist1, nodelist2):
    # generate node dictionary for the two input binaries
    nodeDic1 = {}
    nodeDic2 = {}

    for i in range(len(nodelist1)):
        nodeDic1[nodelist1[i]] = i

    for i in range(len(nodelist2)):
        j = i + len(nodelist1)
        nodeDic2[nodelist2[i]] = j

    print("The two binaries have total of {} nodes.".format(len(nodeDic1) + len(nodeDic2)))
    return nodeDic1, nodeDic2


def instrTypeDicGen(nodelist1, nodelist2):
    # count type of instruction for feature vector generation
    mneList = []

    for node in nodelist1:
        if node.block is None:
            continue
        for insn in node.block.capstone.insns:
            mne = insn.mnemonic
            if mne not in mneList:
                mneList.append(mne)

    for node in nodelist2:
        if node.block is None:
            continue
        for insn in node.block.capstone.insns:
            mne = insn.mnemonic
            if mne not in mneList:
                mneList.append(mne)

    mneDic = {}
    for i in range(len(mneList)):
        mneDic[mneList[i]] = i
    print("there are total of {} types of instructions in the two binaries".format(len(mneList)))
    return mneList, mneDic



def offsetStrMappingGen(cfg1, cfg2, binary1, binary2, mneList):
    # count type of constants for feature vector generation
    
    # offsetStrMapping[offset] = strRef.strip()
    offsetStrMapping = {}

    # lists that store all the non-binary functions in bin1 and 2
    externFuncNamesBin1 = []
    externFuncNamesBin2 = []

    for func in cfg1.functions.values():
        if func.binary_name == binary1:
            for offset, strRef in func.string_references(vex_only=True):
                offset = str(offset)
                #offset = str(hex(offset))[:-1]
                if offset not in offsetStrMapping:
                    offsetStrMapping[offset] = ''.join(strRef.split())
        elif func.binary_name not in externFuncNamesBin1:
            externFuncNamesBin1.append(func.name)
    
    for func in cfg2.functions.values():
        if func.binary_name == binary2:
            for offset, strRef in func.string_references(vex_only=True):
                offset = str(offset)
                #offset = str(hex(offset))[:-1] #[:-1] is to remove the L from say "0x420200L"
                if offset not in offsetStrMapping:
                    offsetStrMapping[offset] = ''.join(strRef.split())
        elif func.binary_name not in externFuncNamesBin2:
            externFuncNamesBin2.append(func.name)
    # constDic = {}
    # i = len(mneList)
    # for key in offsetStrMapping.values():
    #     print('{}: {}'.format(i,key))
    #     constDic[key] = i
    #     i = i + 1

    print("there are total of {} types of strings in the two binaries".format(len(offsetStrMapping)))
    return offsetStrMapping, externFuncNamesBin1, externFuncNamesBin2



# This func extracts the blocks that represent the same external function from both binary 1 and 2. 
# For example, from libc.so
# Somehow angr will create a block in binary 1 and 2 if they call an external function
def externBlocksAndFuncsToBeMerged(cfg1, cfg2, nodelist1, nodelist2, binary1, binary2, nodeDic1, nodeDic2, externFuncNamesBin1, externFuncNamesBin2, string_bid1, string_bid2):
    # toBeMerged[node1_id] = node2_id
    toBeMergedBlocks = {}
    toBeMergedBlocksReverse = {}

    # toBeMergedFuncs[func1_addr] = func2_addr
    toBeMergedFuncs = {}
    toBeMergedFuncsReverse = {}
    
    externFuncNameBlockMappingBin1 = {}
    externFuncNameBlockMappingBin2 = {}
    funcNameAddrMappingBin1 = {}
    funcNameAddrMappingBin2 = {}

    for func in cfg1.functions.values():
        binName = func.binary_name
        funcName = func.name
        funcAddr = func.addr
        blockList = list(func.blocks)
        if (binName == binary1) and (funcName in externFuncNamesBin1) and (len(blockList) == 1):
            for node in nodelist1:
                if (node.block is not None) and (node.block.addr == blockList[0].addr):     
                    externFuncNameBlockMappingBin1[funcName] = nodeDic1[node]
                    funcNameAddrMappingBin1[funcName] = funcAddr

    for func in cfg2.functions.values():
        binName = func.binary_name
        funcName = func.name
        funcAddr = func.addr
        blockList = list(func.blocks)
        if (binName == binary2) and (funcName in externFuncNamesBin2) and (len(blockList) == 1):
            for node in nodelist2:
                if (node.block is not None) and (node.block.addr == blockList[0].addr):     
                    externFuncNameBlockMappingBin2[funcName] = nodeDic2[node]
                    funcNameAddrMappingBin2[funcName] = funcAddr


    for funcName in externFuncNameBlockMappingBin1:
        if funcName in externFuncNameBlockMappingBin2:
            blockBin1 = externFuncNameBlockMappingBin1[funcName]
            blockBin2 = externFuncNameBlockMappingBin2[funcName]
            toBeMergedBlocks[blockBin1] = blockBin2
            toBeMergedBlocksReverse[blockBin2] = blockBin1
            
            func1Addr = funcNameAddrMappingBin1[funcName]
            func2Addr = funcNameAddrMappingBin2[funcName]
            toBeMergedFuncs[func1Addr] = func2Addr
            toBeMergedFuncsReverse[func2Addr] = func1Addr


    # now we also consider string as an indicator for merging
    for opstr in string_bid1:
        if opstr in string_bid2 and len(opstr) > 5:
            bid1 = string_bid1[opstr]
            bid2 = string_bid2[opstr]

            if bid1 in toBeMergedBlocks and bid2 != toBeMergedBlocks[bid1]:
                print("wierd!", bid1, toBeMergedBlocks[bid1], bid2)
            else:
                toBeMergedBlocks[bid1] = bid2
    


    print("TOBEMEGERED size: ", len(toBeMergedBlocks),"\n", toBeMergedBlocks, "\n")
    #print("to be merged funcs: ", toBeMergedFuncs)
    return toBeMergedBlocks, toBeMergedBlocksReverse, toBeMergedFuncs, toBeMergedFuncsReverse




def normalization(opstr, offsetStrMapping):
    optoken = ''

    opstrNum = ""
    if opstr.startswith("0x") or opstr.startswith("0X"):
        opstrNum = str(int(opstr, 16))

    # normalize ptr
    if "ptr" in opstr:
        optoken = 'ptr'
        # nodeToIndex.write("ptr\n")
    # substitude offset with strings
    elif opstrNum in offsetStrMapping:
        optoken = offsetStrMapping[opstrNum]
        # nodeToIndex.write("str\n")
        # nodeToIndex.write(offsetStrMapping[opstr] + "\n")
    elif opstr.startswith("0x") or opstr.startswith("-0x") or opstr.replace('.','',1).replace('-','',1).isdigit():
        optoken = 'imme'
        # nodeToIndex.write("IMME\n")
    elif opstr in register_list_1_byte:
        optoken = 'reg1'
    elif opstr in register_list_2_byte:
        optoken = 'reg2'
    elif opstr in register_list_4_byte:
        optoken = 'reg4'
    elif opstr in register_list_8_byte:
        optoken = 'reg8'
    else:
        optoken = str(opstr)
        # nodeToIndex.write(opstr + "\n")
    return optoken


def nodeIndexToCodeGen(nodelist1, nodelist2, nodeDic1, nodeDic2, offsetStrMapping, outputDir):

    # this dictionary stores the string to block id mapping
    # string_bid[string] = bid
    string_bid1 = {}
    string_bid2 = {}

    # stores the index of block to its tokens
    # blockIdxToTokens[id] = token list of that block
    blockIdxToTokens = {}


    # used to calculate TF part of the TF-IDF
    # it stores # of instructions per block
    blockIdxToOpcodeNum = {}

    # it stores # of instruction appears in one block
    blockIdxToOpcodeCounts = {}

    # calculate IDF part of the information. It stores # of blocks that contain each instruction
    insToBlockCounts = {}


    # store the node index to code mapping for reference
    with open(outputDir + 'nodeIndexToCode', 'w') as nodeToIndex:
        nodeToIndex.write(str(len(nodelist1)) + ' ' + str(len(nodelist2)) + '\n') # write #nodes in both binaries
        
        for node in nodelist1:

            # extract predecessors and successors
            preds = node.predecessors
            succs = node.successors
            preds_ids = []
            succs_ids = []

            for pred in preds:
                preds_ids.append(nodeDic1[pred])
            for succ in succs:
                succs_ids.append(nodeDic1[succ])
            neighbors = [preds_ids, succs_ids]
            per_block_neighbors_bids[nodeDic1[node]] = neighbors


            # go through each instruction to extract token information
            if node.block is None:
                non_code_block_ids.append(nodeDic1[node])
                blockIdxToTokens[str(nodeDic1[node])] = []
                blockIdxToOpcodeCounts[str(nodeDic1[node])] = {}
                blockIdxToOpcodeNum[str(nodeDic1[node])] = 0
                #blockIdxToInstructions[str(nodeDic1[node])] = []
                continue
            tokens = []
            opcodeCounts = {}
            nodeToIndex.write(str(nodeDic1[node]) + ':\n')
            nodeToIndex.write(str(node.block.capstone.insns) + "\n\n")

            # stores the instructions that have been counted for at least once in this block
            countedInsns = []
            numInsns = 0
            for insn in node.block.capstone.insns:
                numInsns = numInsns + 1

                if insn.mnemonic not in opcode_list:
                    opcode_list.append(insn.mnemonic)

                if insn.mnemonic not in countedInsns:
                    if insn.mnemonic not in insToBlockCounts:
                        insToBlockCounts[insn.mnemonic] = 1
                    else:
                        insToBlockCounts[insn.mnemonic] = insToBlockCounts[insn.mnemonic] + 1
                    countedInsns.append(insn.mnemonic)

                if insn.mnemonic not in opcodeCounts:
                    opcodeCounts[insn.mnemonic] = 1
                else:
                    opcodeCounts[insn.mnemonic] = opcodeCounts[insn.mnemonic] + 1

                tokens.append(str(insn.mnemonic))
                opStrs = insn.op_str.split(", ")
                for opstr in opStrs:
                    optoken = normalization(opstr, offsetStrMapping)
                    if optoken != '':
                        tokens.append(optoken)

                    opstrNum = ""
                    if opstr.startswith("0x") or opstr.startswith("0X"):
                        opstrNum = str(int(opstr, 16))
                    if opstrNum in offsetStrMapping:
                        string_bid1[offsetStrMapping[opstrNum]] = nodeDic1[node]

            # nodeToIndex.write("\ttoken:" + str(tokens) + "\n\n")
            blockIdxToTokens[str(nodeDic1[node])] = tokens
            blockIdxToOpcodeCounts[str(nodeDic1[node])] = opcodeCounts
            blockIdxToOpcodeNum[str(nodeDic1[node])] = numInsns

            #blockIdxToInstructions[str(nodeDic1[node])] = insns
            # nodeToIndex.write("\n\n")

        for node in nodelist2:

            # extract predecessors and successors
            preds = node.predecessors
            succs = node.successors
            preds_ids = []
            succs_ids = []

            for pred in preds:
                preds_ids.append(nodeDic2[pred])
            for succ in succs:
                succs_ids.append(nodeDic2[succ])
            neighbors = [preds_ids, succs_ids]
            per_block_neighbors_bids[nodeDic2[node]] = neighbors


            # go through each instruction to extract token information
            if node.block is None:
                non_code_block_ids.append(nodeDic2[node])
                blockIdxToTokens[str(nodeDic2[node])] = []
                blockIdxToOpcodeCounts[str(nodeDic2[node])] = {}
                blockIdxToOpcodeNum[str(nodeDic2[node])] = 0
                continue


            tokens = []
            opcodeCounts = {}
            nodeToIndex.write(str(nodeDic2[node]) + ':\n')
            nodeToIndex.write(str(node.block.capstone.insns) + "\n\n")

            countedInsns = []
            numInsns = 0
            for insn in node.block.capstone.insns:
                numInsns = numInsns + 1

                if insn.mnemonic not in opcode_list:
                    opcode_list.append(insn.mnemonic)

                if insn.mnemonic not in countedInsns:
                    if insn.mnemonic not in insToBlockCounts:
                        insToBlockCounts[insn.mnemonic] = 1
                    else:
                        insToBlockCounts[insn.mnemonic] = insToBlockCounts[insn.mnemonic] + 1
                    countedInsns.append(insn.mnemonic)

                if insn.mnemonic not in opcodeCounts:
                    opcodeCounts[insn.mnemonic] = 1
                else:
                    opcodeCounts[insn.mnemonic] = opcodeCounts[insn.mnemonic] + 1

                tokens.append(str(insn.mnemonic))
                opStrs = insn.op_str.split(", ")
                for opstr in opStrs:
                    optoken = normalization(opstr, offsetStrMapping)
                    if optoken != '':
                        tokens.append(optoken)
                    
                    opstrNum = ""
                    if opstr.startswith("0x") or opstr.startswith("0X"):
                        opstrNum = str(int(opstr, 16))
                    if opstrNum in offsetStrMapping:
                        string_bid2[offsetStrMapping[opstrNum]] = nodeDic2[node]

            # nodeToIndex.write("\ttoken" + str(tokens) + "\n\n")
            blockIdxToTokens[str(nodeDic2[node])] = tokens
            blockIdxToOpcodeCounts[str(nodeDic2[node])] = opcodeCounts
            blockIdxToOpcodeNum[str(nodeDic2[node])] = numInsns
            # nodeToIndex.write("\n\n")

    return blockIdxToTokens, blockIdxToOpcodeNum, blockIdxToOpcodeCounts, insToBlockCounts, string_bid1, string_bid2



def functionIndexToCodeGen(cfg1, cg1, nodelist1, nodeDic1, cfg2, cg2, nodelist2, nodeDic2, binary1, binary2, outputDir):
    # store function addresses
    funclist1 = []
    funclist2 = []
    with open(outputDir + 'functionIndexToCode', 'w') as f:
        f.write(str(len(list(cg1.nodes))) + ' ' + str(len(list(cg2.nodes))) + '\n') # write #nodes in both binaries
        for idx, func in enumerate(list(cg1.nodes)):
            function = cfg1.functions.function(func)

            funclist1.append(function.addr)
            f.write(str(idx) + ':' + '\n')

            f.write('Bin1 ' + function.name + ' ' + hex(function.addr) + ' ' + function.binary_name + '\n')
            for block in function.blocks:
                for node in nodelist1:
                    if (node.block is not None) and (node.block.addr == block.addr):
                        f.write(str(nodeDic1[node]) + ' ')
            f.write('\n')

        for idx, func in enumerate(list(cg2.nodes)):
            function = cfg2.functions.function(func)

            funclist2.append(function.addr)
            f.write(str(idx+len(cg1.nodes)) + ':' + '\n')
            f.write('Bin2 ' + function.name + ' ' + hex(function.addr) + ' ' + function.binary_name +  '\n')
            for block in function.blocks:
                for node in nodelist2:
                    if (node.block is not None) and (node.block.addr == block.addr):
                        f.write(str(nodeDic2[node]) + ' ')
            f.write('\n')
    return funclist1, funclist2



# This function generates super CFG edge list. We also replace external function blocks in binary 2 from block in binary 1
def edgeListGen(edgelist1, nodeDic1, edgelist2, nodeDic2, toBeMerged, toBeMergedReverse, outputDir):
    with open(outputDir + 'edgelist_merged_tadw', 'w') as edgelistFile:
        for (src, tgt) in edgelist1:
            edgelistFile.write(str(nodeDic1[src]) + " " + str(nodeDic1[tgt]) + "\n")
        for (src, tgt) in edgelist2:
            src_id = nodeDic2[src]
            tgt_id = nodeDic2[tgt]

            new_src_id = src_id
            new_tgt_id = tgt_id

            if src_id in toBeMergedReverse:
                new_src_id = toBeMergedReverse[src_id]
            if tgt_id in toBeMergedReverse:
                new_tgt_id = toBeMergedReverse[tgt_id]

            edgelistFile.write(str(new_src_id) + " " + str(new_tgt_id) + "\n")

    with open(outputDir + 'edgelist', 'w') as edgelistFile:
        for (src, tgt) in edgelist1:
            edgelistFile.write(str(nodeDic1[src]) + " " + str(nodeDic1[tgt]) + "\n")
        for (src, tgt) in edgelist2:
            edgelistFile.write(str(nodeDic2[src]) + " " + str(nodeDic2[tgt]) + "\n")


def funcedgeListGen(cg1, funclist1, cg2, funclist2, toBeMergedFuncsReverse, outputDir):
    with open(outputDir + 'func_edgelist', "w") as f:
        for edge in list(cg1.edges):
            f.write(str(funclist1.index(edge[0])) + ' ' + str(funclist1.index(edge[1])) + '\n')
        for edge in list(cg2.edges):
            src_addr = edge[0]
            tgt_addr = edge[1]

            src_id = funclist2.index(src_addr) + len(cg1.nodes)
            tgt_id = funclist2.index(tgt_addr) + len(cg1.nodes)

            new_src_id = src_id
            new_tgt_id = tgt_id

            if src_addr in toBeMergedFuncsReverse:
                new_src_id = funclist1.index(toBeMergedFuncsReverse[src_addr])
            if tgt_addr in toBeMergedFuncsReverse:
                new_tgt_id = funclist1.index(toBeMergedFuncsReverse[tgt_addr])

            f.write(str(new_src_id) + ' ' + str(new_tgt_id) + '\n')


# not used. we now generate node features from asm2vec
def nodeFeaturesGen(nodelist1, nodelist2, mneList, mneDic, constDic, offsetStrMapping, outputDir):
    # generate feature vector file for the two input binaries
    with open(outputDir + 'features','w') as feaVecFile:
        for i in range(len(nodelist1)):
            node = nodelist1[i]
            feaVec = []
            for _ in range(len(mneList) + len(offsetStrMapping)):
                feaVec.append(0)
            if node.block is not None:
                for const in node.block.vex.constants:
                    if str(const) != 'nan':
                        offset = str(const.value)#hex(int(const.value))
                    if offset in offsetStrMapping:
                        c = offsetStrMapping.get(offset)
                        pos = constDic[c]
                        feaVec[pos] += 1

                for insn in node.block.capstone.insns:
                    mne = insn.mnemonic
                    pos = mneDic[mne]
                    feaVec[pos] += 1

            # index as the first element and then output all the features
            feaVecFile.write(str(i) + " ")
            for k in range(len(feaVec)):
                feaVecFile.write(str(feaVec[k]) + " ")
            feaVecFile.write("\n")

        for i in range(len(nodelist2)):
            node = nodelist2[i]
            feaVec = []
            for x in range(len(mneList) + len(offsetStrMapping)):
                feaVec.append(0)
            if node.block is not None:
                for const in node.block.vex.constants:
                    if str(const) != 'nan':
                        offset = str(const.value)#hex(int(const.value))
                    if offset in offsetStrMapping:
                        c = offsetStrMapping.get(offset)
                        pos = constDic[c]
                        feaVec[pos] += 1
                        
                for insn in node.block.capstone.insns:
                    mne = insn.mnemonic
                    pos = mneDic[mne]
                    feaVec[pos] += 1
            j = i + len(nodelist1)
            feaVecFile.write(str(j) + " ")
            for k in range(len(feaVec)):
                feaVecFile.write(str(feaVec[k]) + " ")
            feaVecFile.write("\n")



# preprocessing the two binaries with Angr. 
def preprocessing(filepath1, filepath2, outputDir):
        
        binary1 = path_leaf(filepath1)
        binary2 = path_leaf(filepath2)

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        cfg1, cg1, nodelist1, edgelist1, cfg2, cg2, nodelist2, edgelist2 = angrGraphGen(filepath1, filepath2)
        nodeDic1, nodeDic2 = nodeDicGen(nodelist1, nodelist2)

        mneList, _ = instrTypeDicGen(nodelist1, nodelist2)

        # print("\t extracing strings...")
        offsetStrMapping, externFuncNamesBin1, externFuncNamesBin2 = offsetStrMappingGen(cfg1, cfg2, binary1, binary2, mneList)
        
        print("\tprocessing instructions...")
        blockIdxToTokens, blockIdxToOpcodeNum, blockIdxToOpcodeCounts, insToBlockCounts, string_bid1, string_bid2 = nodeIndexToCodeGen(nodelist1, nodelist2, nodeDic1, nodeDic2, offsetStrMapping, outputDir)

        toBeMergedBlocks, toBeMergedBlocksReverse, toBeMergedFuncs, toBeMergedFuncsReverse = externBlocksAndFuncsToBeMerged(cfg1, cfg2, nodelist1, nodelist2, binary1, binary2, nodeDic1, nodeDic2, externFuncNamesBin1, externFuncNamesBin2, string_bid1, string_bid2)
        
        # print("\t processing functions...")
        # funclist1, funclist2 = functionIndexToCodeGen(cfg1, cg1, nodelist1, nodeDic1, cfg2, cg2, nodelist2, nodeDic2, binary1, binary2, outputDir)

        print("\tgenerating CFGs...")
        edgeListGen(edgelist1, nodeDic1, edgelist2, nodeDic2, toBeMergedBlocks, toBeMergedBlocksReverse, outputDir)

        # print("\t generating call graphs...")
        # funcedgeListGen(cg1, funclist1, cg2, funclist2, toBeMergedFuncsReverse, outputDir)

        print("Preprocessing all done. Enjoy!!")
        return blockIdxToTokens, blockIdxToOpcodeNum, blockIdxToOpcodeCounts, insToBlockCounts, nodeDic1, nodeDic2, binary1, binary2, toBeMergedBlocks