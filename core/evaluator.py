import logging
import random

from abc import ABC, abstractmethod
from typing import List
from core.gen_models import GenerationModel


logger = logging.getLogger(__name__)


class RespRanker(ABC):
	@abstractmethod
	def evaluate(self, context, resp_a, resp_b):
		"""
		Compare two responses and return the preference.
		"""
		raise NotImplementedError


class P4GEvaluator(RespRanker):
	def __init__(self, gen_model: GenerationModel):
		super().__init__()
		self.gen_model = gen_model
		self.inference_args = {
			"max_tokens": 2,
			"temperature": 0.7,
			"echo": False,
			"n": 5,
			"stop": ""
		}
	
	def evaluate(self, context, resp_a, resp_b):
		do_swap = False
		if random.random() < 0.5:
			do_swap = True
			resp_a, resp_b = resp_b, resp_a
		prompt = f"""
		The following is background information about Save the Children.
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		The following is a conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		{context}
		Which of the following responses can better help the Persuader convince the Persuadee to donate to Save the Children?
		A. Persuader: {resp_a}
		B. Persuader: {resp_b}
		C. Can't tell.
		Your can choose from either A, B, or C.
		Your choice:
		""".replace('\t', '').strip()
		logger.debug(f"prompt: {prompt}")
		resps = self.gen_model.generate(prompt, **self.inference_args)
		choices, rationales = self._process_resps(resps)
		preference = self._majority_vote(choices, do_swap)
		return preference, {'choices': choices, 'rationales': rationales, 'do_swap': do_swap}

	def _process_resps(self, resps:List[dict]):
		choices = []
		rationales = []
		for resp in resps:
			gen = resp['generated_text'].strip()
			
			if len(gen) == 0:
				print("Empty response")
				choice = 'c'
			else:
				choice = gen[0].lower()
			
			if choice not in ['a', 'b', 'c']:
				print(f"Invalid choice: {choice}")
				choice = 'c'
			choices.append(choice)
			# see if there is a rationale  # just dump the entire response
			rationale = gen
			rationales.append(rationale)
		return choices, rationales

	def _majority_vote(self, resps:List[str], do_swap=False):
		# if there is a majority vote between A=0 and B=1, return the majority vote
		# otherwise, return C=2
		a_cnt = 0
		b_cnt = 0
		for resp in resps:
			if resp == 'a':
				a_cnt += 1
			elif resp == 'b':
				b_cnt += 1
		if a_cnt > b_cnt:
			return 0 if not do_swap else 1
		elif b_cnt > a_cnt:
			return 1 if not do_swap else 0
		return 2