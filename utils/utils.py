import torch
import random
import numpy as np

def set_determinitic_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	return


class dotdict(dict):
	def __getattr__(self, name):
		return self[name]


class hashabledict(dict):
	def __hash__(self):
		return hash(tuple(sorted(self.items())))