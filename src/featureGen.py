import collections
import math
import os
import random

import numpy as np
import tensorflow as tf

import deepbindiff

# CONFIGURATION
######################################################################################
batch_size = 128
embedding_size = 64  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

#num_steps = 100001
prev_data_index = 0
data_index = 0

random_walk_done = False
#####################################################################################

debug = False

# find the neighboring instructions based on current data_index
# return []
def getNeighboringInsn(article, opcode_idx_list, insnStartingIndices, indexToCurrentInsnsStart):
    # data_index will already be an index for block boundary
    # we can get the batch size by simplying check the position of data_index in blockBoundaryIdx
    global data_index
    global debug

    # ib_len = len(insnBoundaries)
    # insnPos = 0
    # # if current token is in the first or last instruction in the block, it only has 1 neibhoring instruction, otherwise 2.
    # # insnPos == 1 means only has one next instruction, 2 means has both prev and next instructions, 3 means ony has prev instruction
    # if data_index < insnBoundaries[0]:
    #     insnPos = 1
    # elif data_index > insnBoundaries[ib_len - 1]:
    #     insnPos = 2
    # else:
    #     insnPos = 3

    
    # # # each minibatch should contain 3 blocks (2 if we hit the beginning block or the ending block)
    # insnNum = 0
    # if data_index == 0:
    #     insnNum = 2
    # else:
    #     for counter, value in enumerate(blockBoundaryIdx):
    #         if value == (data_index - 1):
    #             if counter == (len(blockBoundaryIdx) - 1):
    #                 insnNum = 2
    #             else:
    #                 insnNum = 3
    
    # if insnPos == 0:
    #     print("error in getting batch size")
    #     exit
    
    # temp = data_index
    # print("blockNum: ", insnPos)
    # print("temp: ", temp, article[temp])
    # batch_size = 0
    # for counter, _ in enumerate(article[temp + 1:]):
    #     currentIdx = counter + temp + 1
    #     batch_size = batch_size + 1
    #     if currentIdx in blockBoundaryIdx:
    #         insnPos = insnPos -1
    #     if insnPos == 0:
    #         break
    
    # data_index = data_index + batch_size + 1

    if debug == True:
        print("current data_index: ", data_index)
    
    # find the insn boundary for current token
    currentInsnStart = -1
    
    if data_index in indexToCurrentInsnsStart:
        currentInsnStart = indexToCurrentInsnsStart[data_index]
    else:
        if data_index <= len(article) - 1:
            currentInsnStart = len(insnStartingIndices) - 1
        else:
            print("error in getting current instruction starting index!")
            raise SystemExit

    if debug == True:
        print("currentInsnStart: ", currentInsnStart)

    return currentInsnStart
        




# generate minibatches for a given random walk.
# consider one instruction before and one instruction after as the context for each token 
def generate_batch(article, blockBoundaryIdx, insnStartingIndices, indexToCurrentInsnsStart, opcode_idx_list):
    global data_index
    global batch_size
    global random_walk_done

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    article_size = len(article)
    target = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # each instructin can have almost one opcode and three operands
    # labels_new[i, 0, 0], labels_new[i, 1, 0] are the numbers of token in the instruction

    context = np.ndarray(shape=(batch_size, 2, 5), dtype=np.int32)
    context = np.full((batch_size, 2, 5), -1)
    
    
    # labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # span = 2 * skip_window + 1  # [ skip_window target skip_window ]

    # # If maxlen is not specified or is None, deques may grow to an arbitrary length. Otherwise, the deque is bounded to the specified maximum length. Once a bounded length deque is full, when new items are added, a corresponding number of items are discarded from the opposite end
    # buffer = collections.deque(maxlen=span)
    # for _ in range(span):
    #     buffer.append(article[data_index])
    #     data_index = (data_index + 1) % datalen

    # for i in range(batch_size // num_skips):
    #     target = skip_window  # target label at the center of the buffer
    #     targets_to_avoid = [skip_window]
    #     for j in range(num_skips):
    #         while target in targets_to_avoid:
    #             target = random.randint(0, span - 1)
    #         targets_to_avoid.append(target)
    #         batch[i * num_skips + j] = buffer[skip_window]
    #         # labels[i * num_skips + j, 0] = buffer[target]
    #     buffer.append(article[data_index])
    #     data_index = (data_index + 1) % datalen

    # # Backtrack a little bit to avoid skipping words in the end of a batch
    # data_index = (data_index + datalen - span) % datalen



    for i in range(batch_size):
        currentInsnStart = getNeighboringInsn(article, opcode_idx_list, insnStartingIndices, indexToCurrentInsnsStart)
        target[i, 0] = article[data_index]

        prevInsnStart = -1
        prevInsnEnd = -1
        nextInsnStart = -1
        nextInsnEnd = -1

        if article_size != 1 and len(insnStartingIndices) > 1:
            if currentInsnStart == 0:
                nextInsnStart = insnStartingIndices[currentInsnStart + 1]
            elif currentInsnStart ==  (len(insnStartingIndices) -1):
                prevInsnStart = insnStartingIndices[currentInsnStart - 1]
            else:
                prevInsnStart = insnStartingIndices[currentInsnStart - 1]
                nextInsnStart = insnStartingIndices[currentInsnStart + 1]
            
            if prevInsnStart != -1:
                prevInsnEnd = insnStartingIndices[currentInsnStart] - 1


            if nextInsnStart != -1:
                if nextInsnStart == insnStartingIndices[len(insnStartingIndices) - 1]:
                    nextInsnEnd = len(article) - 1
                else:
                    nextInsnEnd = insnStartingIndices[currentInsnStart + 2] - 1

        if debug == True:
            print("prevInsnStart: ", prevInsnStart)
            print("prevInsnEnd: ", prevInsnEnd)
            print("nextInsnStart: ", nextInsnStart)
            print("nextInsnEnd: ", nextInsnEnd, "\n")

        next_index = (data_index + 1) % article_size

        if next_index <= data_index:
            random_walk_done = True
        data_index = next_index

        if prevInsnStart != -1:
            # should be + 1, + 2 to make embedding_lookup easier
            context[i, 0, 0] = prevInsnEnd + 2 - prevInsnStart
            for j in range(context[i, 0, 0] - 1):
                context[i, 0, j + 1]  = article[prevInsnStart + j]
        else:
            context[i, 0, 0] = 0

        if nextInsnStart != -1:
            context[i, 1, 0] = nextInsnEnd + 2 - nextInsnStart            
            for j in range(context[i, 1, 0] - 1):
                context[i, 1, j + 1]  = article[nextInsnStart + j]
        else:
            context[i, 1, 0] = 0


    return context, target 


def get_insns_token_embeddings(embeddings, insn):
    return tf.nn.embedding_lookup(embeddings, insn[1:insn[0]])


# def zero_embedding():
#     return tf.zeros([embedding_size])


# def random_embedding():
#     return tf.random_uniform([embedding_size], -1.0, 1.0)


def cal_operand_embedding(insnToken_embeddings, insn_opcode):
    size = tf.to_float(tf.size(insnToken_embeddings))
    operand_embedding = tf.subtract(tf.div(tf.reduce_sum(insnToken_embeddings, 0), size), tf.div(insn_opcode, size))
    return operand_embedding


def cal_insn_embedding(embeddings, insn, insn_size):
    insnToken_embeddings = get_insns_token_embeddings(embeddings, insn)
    insn_opcode = insnToken_embeddings[0]

    has_no_operand = tf.equal(insn_size, tf.Variable(1))
    insn_operand = tf.cond(has_no_operand, 
        true_fn=lambda: tf.zeros([embedding_size]), 
        false_fn=lambda: cal_operand_embedding(insnToken_embeddings, insn_opcode))

    insn_embedding = tf.concat([insn_opcode, insn_operand], 0)
    # insn_embedding = tf.add(insn_opcode, insn_operand)
    return insn_embedding





def buildAndTraining(article, blockBoundaryIndex, insnStartingIndices, indexToCurrentInsnsStart, dictionary, opcode_idx_list):
    global data_index
    global random_walk_done

    dic_size = len(dictionary)

    g = tf.Graph()
    with g.as_default():
    # Input data.
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2, 5])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([dic_size, embedding_size], -1.0, 1.0))

            # TODO: needs to be reinitilized for every random walk
            # rw_embed = tf.Variable(tf.random_uniform([1, 2 * embedding_size], -1.0, 1.0))
            # token_embeddings = tf.nn.embedding_lookup(embeddings, train_inputs)
            temp_embeddings = tf.zeros([1, 2 * embedding_size])

            
            # now, prevInsns and nextInsns are both [batch_size, 1, 5] after the split
            # and become [batch_size, 5] after the squeeze
            prevInsns, nextInsns = tf.split(train_inputs, num_or_size_splits=2, axis=1)
            prevInsns = tf.squeeze(prevInsns)
            nextInsns = tf.squeeze(nextInsns)

            for i in range(batch_size):
                prevInsn = prevInsns[i]
                nextInsn = nextInsns[i]
                prevInsn_size = prevInsn[0]
                nextInsn_size = nextInsn[0]

                has_prev = tf.not_equal(prevInsn_size, tf.Variable(0))
                has_next = tf.not_equal(nextInsn_size, tf.Variable(0))
                prevInsn_embedding = tf.cond(has_prev, 
                    lambda: cal_insn_embedding(embeddings, prevInsn, prevInsn_size), 
                    lambda: tf.random_uniform([2 * embedding_size], -1.0, 1.0))
                nextInsn_embedding = tf.cond(has_next, 
                    lambda: cal_insn_embedding(embeddings, nextInsn, nextInsn_size), 
                    lambda: tf.random_uniform([2 * embedding_size], -1.0, 1.0))

                
                # currInsn = tf.div(tf.add(tf.add(prevInsn_embedding, nextInsn_embedding), rw_embed), 3.0)
                currInsn = tf.div(tf.add(prevInsn_embedding, nextInsn_embedding), 2.0)
                currInsn = tf.reshape(currInsn, [1, 2 * embedding_size])
                temp_embeddings = tf.concat([temp_embeddings, currInsn], 0)

            insn_embeddings = tf.slice(temp_embeddings, [1, 0], [batch_size, 2 * embedding_size])

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(tf.truncated_normal([dic_size, 2 * embedding_size], stddev=1.0 / math.sqrt(2 * embedding_size)))
            nce_biases = tf.Variable(tf.zeros([dic_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                biases=nce_biases,
                                labels=train_labels,
                                inputs=insn_embeddings,
                                num_sampled=num_sampled,
                                num_classes=dic_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        # valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        # similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()


    # training
    with tf.Session(graph=g) as session:
        # We must initialize all variables before we use them.
        init.run(session=session)
        print('Initialized')

        # rw_embedding_list = []
        # for each random walk, we learn an embedding for the random walk and for all the blocks in the walk
        data_index = 0
        num_steps = 6001

        average_loss = 0
        for step in range(num_steps):
            context, target = generate_batch(article, blockBoundaryIndex, insnStartingIndices, indexToCurrentInsnsStart, opcode_idx_list)
            feed_dict = {train_inputs: context, train_labels: target}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 500 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0
            
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            # if step % 2000 == 0:
            #     sim = similarity.eval()
            #     for i in range(valid_size):
            #         # dictionary: index to token
            #         valid_word = dictionary[valid_examples[i]]
            #         top_k = 8  # number of nearest neighbors
            #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            #         log_str = 'Nearest to %s:' % valid_word
            #         for k in range(top_k):
            #             close_word = dictionary[nearest[k]]
            #             log_str = '%s %s,' % (log_str, close_word)
            #         print(log_str)


            # no point reusing the same training data to train again
            # stop the current random walk if we have used all of it for training
            # if random_walk_done:
            #     random_walk_done = False
            #     break


            # rw_embedding_list.append(rw_embed.eval())
            # print("Randon walk", counter, " embedding: ", rw_embed.eval())

        final_embeddings = normalized_embeddings.eval()



    return final_embeddings



def tokenEmbeddingGeneration(article, blockBoundaryIndex, insnStartingIndices, indexToCurrentInsnsStart, dictionary, reversed_dictionary, opcode_idx_list):
    embeddings = buildAndTraining(article, blockBoundaryIndex, insnStartingIndices, indexToCurrentInsnsStart, dictionary, opcode_idx_list)
    return embeddings
