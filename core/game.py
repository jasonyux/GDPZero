import numpy as np
import logging

from core.gen_models import DialogModel
from core.helpers import DialogSession
from abc import ABC, abstractmethod
from typing import List


logger = logging.getLogger(__name__)


class DialogGame(ABC):
	def __init__(self, 
			system_name:str, system_agent:DialogModel, 
			user_name: str, user_agent:DialogModel):
		self.SYS = system_name
		self.system_agent = system_agent
		self.USR = user_name
		self.user_agent = user_agent
		return

	@staticmethod
	@abstractmethod
	def get_game_ontology() -> dict:
		"""returns game related information such as dialog acts, slots, etc.
		"""
		raise NotImplementedError

	def init_dialog(self) -> DialogSession:
		# [(sys_act, sys_utt, user_act, user_utt), ...]
		return DialogSession(self.SYS, self.USR)

	def get_next_state(self, state:DialogSession, action) -> DialogSession:
		next_state = state.copy()

		sys_utt = self.system_agent.get_utterance(next_state, action)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		next_state.add_single(state.SYS, sys_da, sys_utt)
		
		# state in user's perspective
		user_da, user_resp = self.user_agent.get_utterance_w_da(next_state, None)  # user just reply
		next_state.add_single(state.USR, user_da, user_resp)
		return next_state
	
	def get_next_state_batched(self, state:DialogSession, action, batch=3) -> List[DialogSession]:
		all_next_states = [state.copy() for _ in range(batch)]

		sys_utts = self.system_agent.get_utterance_batched(state.copy(), action, batch)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		for i in range(batch):
			all_next_states[i].add_single(state.SYS, sys_da, sys_utts[i])
		
		# state in user's perspective
		user_das, user_resps = self.user_agent.get_utterance_w_da_from_batched_states(all_next_states, None)  # user just reply
		for i in range(batch):
			all_next_states[i].add_single(state.USR, user_das[i], user_resps[i])
		return all_next_states

	def display(self, state:DialogSession):
		string_rep = state.to_string_rep(keep_sys_da=True, keep_user_da=True)
		print(string_rep)
		return

	@abstractmethod
	def get_dialog_ended(self, state) -> float:
		"""returns 0 if not ended, then (in general) 1 if system success, -1 if failure 
		"""
		raise NotImplementedError


class PersuasionGame(DialogGame):
	SYS = "Persuader"
	USR = "Persuadee"

	S_PersonalStory = "personal story"
	S_CredibilityAppeal = "credibility appeal"
	S_EmotionAppeal = "emotion appeal"
	S_PropositionOfDonation = "proposition of donation"
	S_FootInTheDoor = "foot in the door"
	S_LogicalAppeal = "logical appeal"
	S_SelfModeling = "self modeling"
	S_TaskRelatedInquiry = "task related inquiry"
	S_SourceRelatedInquiry = "source related inquiry"
	S_PersonalRelatedInquiry = "personal related inquiry"
	S_NeutralToInquiry = "neutral to inquiry"
	S_Greeting = "greeting"
	S_Other = "other"

	U_NoDonation = "no donation"
	U_NegativeReaction = "negative reaction"
	U_Neutral = "neutral"
	U_PositiveReaction = "positive reaction"
	U_Donate = "donate"

	def __init__(self, system_agent:DialogModel, user_agent:DialogModel, 
			max_conv_turns=15):
		super().__init__(PersuasionGame.SYS, system_agent, PersuasionGame.USR, user_agent)
		self.max_conv_turns = max_conv_turns
		return

	@staticmethod
	def get_game_ontology() -> dict:
		return {
			"system": {
				"dialog_acts": [
					PersuasionGame.S_PersonalStory, PersuasionGame.S_CredibilityAppeal, PersuasionGame.S_EmotionAppeal,
					PersuasionGame.S_PropositionOfDonation, PersuasionGame.S_FootInTheDoor, PersuasionGame.S_LogicalAppeal,
					PersuasionGame.S_SelfModeling, PersuasionGame.S_TaskRelatedInquiry, PersuasionGame.S_SourceRelatedInquiry,
					PersuasionGame.S_PersonalRelatedInquiry, PersuasionGame.S_NeutralToInquiry, PersuasionGame.S_Greeting,
					PersuasionGame.S_Other
				],
			},
			"user": {
				"dialog_acts": [
					PersuasionGame.U_NoDonation, PersuasionGame.U_NegativeReaction, PersuasionGame.U_Neutral,
					PersuasionGame.U_PositiveReaction, PersuasionGame.U_Donate
				]
			}
		}

	def get_dialog_ended(self, state) -> float:
		# terminate if there is a <donate> action in persudee resp
		# allow only 10 turns
		if len(state) >= self.max_conv_turns:
			logger.info("Dialog ended with persuasion failure")
			return -1.0
		for (_, da, _) in state:
			if da == PersuasionGame.U_Donate:
				logger.info("Dialog ended with donate")
				return 1.0
			if da == PersuasionGame.U_NoDonation:
				logger.info("Dialog ended with no-donation")
				return -1.0
		return 0.0


class EmotionalSupportGame(PersuasionGame):
	pass