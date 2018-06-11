from os.path import join, dirname
import numpy as np

DATA_DIR = dirname(__file__)

hama_phd_qe = np.loadtxt(join(DATA_DIR, "phdqe.txt"))
