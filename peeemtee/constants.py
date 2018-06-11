from os.path import join, dirname
import numpy as np

DATA_DIR = join(dirname(__file__), "data")

hama_phd_qe = np.loadtxt(join(DATA_DIR, "phdqe.txt"))
