import idaapi
import idautils
import idc
import re

bin_name = idc.GetInputFile()
edge_list_file = bin_name + "_edgelist.txt"
features_file = bin_name + ".features"

funcs = list(idautils.Functions())

idc.GenCallGdl(bin_name, 'Call Gdl', idc.CHART_GEN_GDL)
node_label = []
node_name = []
edges = []
# for func in funcs:
with open(bin_name + ".gdl") as f:
    for line in f:
        if not line[:2] == '//':
            para_list = line.split()
            if para_list[0] == "node:":
                node_label.append(re.findall('\"(.*)\"', para_list[3])[0])
                node_name.append(re.findall('\"(.*)\"', para_list[5])[0])
            elif para_list[0] == "edge:":
                edges.append([re.findall('\"(.*)\"', para_list[3])[0], re.findall('\"(.*)\"', para_list[5])[0]])

with open(edge_list_file, "w") as f:
    for edge in edges:
        f.write(edge[0] + " " + edge[1] + '\n')  

opcode_list = []

for ea in funcs:
    E = list(FuncItems(ea))
    for e in E:
        opcode = GetDisasm(e).split()[0]
        if opcode not in opcode_list: 
            opcode_list.append(opcode)

# funcs = idautils.Functions()
with open(features_file, "w") as f:
    for ea in funcs:
        blocks = idaapi.FlowChart(idaapi.get_func(ea))
        function_name = GetFunctionName(ea)
        # if function_name in node_name:
        index = node_name.index(function_name)
        feature_vec = [0]* len(opcode_list)
        E = list(FuncItems(ea))
        for e in E:
            opcode = GetDisasm(e).split()[0]
            feature_vec[opcode_list.index(opcode)] += 1
        feature_vec.insert(0, index)        
        for i in feature_vec:
            f.write(str(i) + " ")
        f.write('\n')
        for block in blocks:
            for head in idautils.Heads(block.startEA, block.endEA):
                print function_name, ":", block.id, ":", "0x%08x"%(head), ":", GetDisasm(head)
            for s in block.succs():
                print block.id, "-->", s.id