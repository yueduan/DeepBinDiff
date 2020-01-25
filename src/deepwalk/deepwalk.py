import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

from deepwalk import graph
#from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec
from deepwalk  import skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

# import psutil
# from multiprocessing import cpu_count

# p = psutil.Process(os.getpid())
# try:
#     p.set_cpu_affinity(list(range(cpu_count())))
# except AttributeError:
#     try:
#         p.cpu_affinity(list(range(cpu_count())))
#     except AttributeError:
#         pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


def process(edgelistFile, indexToTokens, undirected, number_walks, walk_length, seed):
  G = graph.load_edgelist(edgelistFile, undirected=undirected)

  print("Number of nodes: {}".format(len(G.nodes())))
  num_walks = len(G.nodes()) * number_walks
  print("Number of walks: {}".format(num_walks))
  data_size = num_walks * walk_length
  print("Data size (walks*length): {}".format(data_size))

  print("Walking...")
  walks = graph.build_deepwalk_corpus(G, num_paths=number_walks, path_length=walk_length, alpha=0, rand=random.Random(seed))
  

  return walks


  #   print("Training...")
  #   model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
  # else:
  #   print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
  #   print("Walking...")

  #   walks_filebase = args.output + ".walks"
  #   walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
  #                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
  #                                        num_workers=args.workers)

  #   print("Counting vertex frequency...")
  #   if not args.vertex_freq_degree:
  #     vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
  #   else:
  #     # use degree distribution for frequency in tree
  #     vertex_counts = G.degree(nodes=G.iterkeys())

  #   print("Training...")
  #   walks_corpus = serialized_walks.WalksCorpus(walk_files)
  #   model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
  #                    size=args.representation_size,
  #                    window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)

  # model.wv.save_word2vec_format(args.output)


# number_walks: number of walks per node
# walk_length: the length of each random walk
# seed: random seed
def randomWalksGen(edgelistFile, indexToTokens, undirected=False, number_walks=2, walk_length=4, seed=0):
  return process(edgelistFile, indexToTokens, undirected, number_walks, walk_length, seed)

# if __name__ == "__main__":
#     print("deepwalk.py")