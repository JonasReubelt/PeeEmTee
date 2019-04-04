from os.path import join, dirname
import numpy as np

DATA_DIR = join(dirname(__file__), "data")

hama_phd_qe = np.loadtxt(join(DATA_DIR, "phdqe.txt"))

thor_data = np.loadtxt(join(DATA_DIR, "phdqe_thorlabs.txt"), unpack=True)
thor_phd_qe = np.array((thor_data[0], thor_data[2])).T
