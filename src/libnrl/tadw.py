from __future__ import print_function
import numpy as np
from sklearn.preprocessing import normalize

import preprocessing

class TADW(object):

    def __init__(self, graph, dim, lamb=0.2, threshold=0.1):
        self.g = graph
        self.lamb = lamb
        self.dim = dim
        self.threshold = threshold
        self.train()

    def getAdj(self):
        node_size = self.g.node_size
        adj = np.zeros((node_size, node_size))
        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
            adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
        # ScaleSimMat
        rowsum = np.sum(adj, axis=1)
        # for i in range(len(rowsum)):
        #     if rowsum[i] == 0.0:
        #         rowsum[i] = 0.001
        adj = adj/rowsum
        return adj

    
    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        # node_num = len(self.vectors.keys())
        # fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            if node not in preprocessing.non_code_block_ids:
                fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        
        # Yue: # put raw feature vector as embedding when nodes are singular nodes
        embedding_len = len(vec)
        extra_feature_dim = embedding_len - len(self.g.G.nodes[node]['feature'])

        # print("preprocessing.non_code_block_ids:", preprocessing.non_code_block_ids)
        # no_embedding_list = self.g.sgl_node_list + preprocessing.non_code_block_ids

        for sgl_node in self.g.sgl_node_list:
        #for sgl_node in no_embedding_list:
            feature_list = []

            if extra_feature_dim > 0:
                for x in self.g.singular_node_dict[sgl_node]:
                    feature_list.append(str(x))
                for _x in range(extra_feature_dim):
                    feature_list.append(str(0))
            else: # handle the case where feature vector is longer than embedding
                for idx, x in enumerate(self.g.singular_node_dict[sgl_node]):
                    if idx < embedding_len:
                        feature_list.append(str(x))

            fout.write("{} {}\n".format(sgl_node, ' '.join([str(x) for x in feature_list])))
        fout.close()

    def getT(self):
        g = self.g.G
        look_back = self.g.look_back_list
        self.features = np.vstack([g.nodes[look_back[i]]['feature']
                                   for i in range(g.number_of_nodes())])
        # we will not use their Dimensionality Reduction function.
        # Their apporach seems have some bug...
        # self.preprocessFeature()
        return self.features.T

    def preprocessFeature(self):
        U, S, VT = np.linalg.svd(self.features)
        Ud = U[:, 0:self.features.shape[1]]
        Sd = S[0:self.features.shape[1]]
        self.features = np.array(Ud)*Sd.reshape(self.features.shape[1])

    def train(self):

        def mat_pow(mat, power):
            assert power > 0
            result = mat
            if power == 1:
                return result
            else:
                for i in range(power-1):
                    result = np.dot(result, mat)
                return result

        self.adj = self.getAdj()
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix TODO need to be enhanced.
        self.M = (self.adj + mat_pow(self.adj, 2))/2
        # T is feature_size*node_num, text features
        self.T = self.getT()
        self.node_size = self.adj.shape[0]
        self.feature_size = self.features.shape[1]
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)

        # Update
        for i in range(20):
            # Update W
            B = np.dot(self.H, self.T)
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - \
                2 * np.dot(B, self.M.T) + self.lamb*self.W
            Hess = 2*np.dot(B, B.T) + self.lamb*np.eye(self.dim)
            drv = np.reshape(drv, [self.dim*self.node_size, 1])
            rt = - drv
            dt = rt
            vecW = np.reshape(self.W, [self.dim*self.node_size, 1])
            while np.linalg.norm(rt, 2) > 3e-2:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [self.dim*self.node_size, 1])
                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                vecW = vecW + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            # Update H
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)
                         - np.dot(self.W, self.M.T)), self.T.T) + self.lamb*self.H
            drv = np.reshape(drv, (self.dim*self.feature_size, 1))
            rt = - drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim*self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 3e-2:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))
                                 + self.lamb*dtS, (self.dim*self.feature_size, 1))
                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                vecH = vecH + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))
            # print(self.W.T.shape, self.H.shape, self.T.shape)
            B = np.dot(self.H, self.T) # B = H x T
            dis = np.linalg.norm(2 * np.dot(np.dot(B, B.T), self.W) - \
                2 * np.dot(B, self.M.T)) + np.linalg.norm(self.lamb*self.W) + np.linalg.norm(self.lamb*self.H)
            if i == 0:
                lst_dis = dis + self.threshold + 1
            if lst_dis - dis < self.threshold:
                break
            lst_dis = dis
        self.Vecs = np.hstack((normalize(self.W.T), normalize(np.dot(self.T.T, self.H.T))))
        # get embeddings
        self.vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self.Vecs):
            self.vectors[look_back[i]] = embedding
