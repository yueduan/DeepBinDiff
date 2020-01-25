from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import normalize

class TADW_GPU(object):

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
        for i in range(len(rowsum)):
            if rowsum[i] == 0.0:
                rowsum[i] = 0.001
        adj = adj/rowsum
        return adj

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        # node_num = len(self.vectors.keys())
        # fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
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
        np.set_printoptions(threshold=np.inf, suppress=True)
        self.adj = self.getAdj()
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix TODO need to be enhanced.      
        self.M = (self.adj + mat_pow(self.adj, 2))/2
        self.T = self.getT()
        self.node_size = self.adj.shape[0]
        self.feature_size = self.features.shape[1]

        temp_W = np.random.randn(self.dim, self.node_size)
        temp_H = np.random.randn(self.dim, self.feature_size)

        self.W = tf.Variable(temp_W, dtype=tf.float64)
        self.H = tf.Variable(temp_H, dtype=tf.float64)
        res_W = [temp_W]
        res_H = [temp_H]
        ph_W = tf.placeholder(tf.float64, shape=res_W[-1].shape)
        ph_H = tf.placeholder(tf.float64, shape=res_H[-1].shape)

        B = tf.matmul(ph_H, self.T)
        loss_W = pow(tf.norm(self.M - tf.matmul(self.W, tf.matmul(ph_H, self.T), transpose_a=True), ord='fro', axis=[0, 1]), 2) + \
                 self.lamb * (pow(tf.norm(self.W, ord='fro', axis=[0, 1]),2) + pow(tf.norm(ph_H, ord='fro', axis=[0, 1]), 2))
        loss_H = pow(tf.norm(self.M - tf.matmul(ph_W, tf.matmul(self.H, self.T), transpose_a=True), ord='fro', axis=[0, 1]), 2) + \
                 self.lamb * (pow(tf.norm(ph_W, ord='fro', axis=[0, 1]), 2) + pow(tf.norm(self.H, ord='fro', axis=[0, 1]), 2))

        train_W = tf.train.AdamOptimizer(0.01).minimize(loss_W)
        train_H = tf.train.AdamOptimizer(0.01).minimize(loss_H)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(100):
                print('Iteration ', i)
                # Update W
                for i in range(50):
                    sess.run(train_W, feed_dict={ph_H:res_H[-1]})
                res_W.append(sess.run(self.W))
                # Update H
                for i in range(50):
                    sess.run(train_H, feed_dict={ph_W:res_W[-1]})
                res_H.append(sess.run(self.H))
                HT = np.dot(res_H[-1], self.T)
                lossfunc = np.linalg.norm(2 * np.dot(np.dot(HT, HT.T), res_W[-1]) - \
                    2 * np.dot(HT, self.M.T)) + np.linalg.norm(self.lamb*res_W[-1]) + np.linalg.norm(self.lamb*res_H[-1])
                print(lossfunc)
            self.Vecs = np.hstack((normalize(temp_W.T), normalize(np.dot(self.T.T, temp_H.T))))
            self.vectors = {}
            look_back = self.g.look_back_list
            for i, embedding in enumerate(self.Vecs):
                self.vectors[look_back[i]] = embedding
