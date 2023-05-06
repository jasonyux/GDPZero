import numpy as np
import logging
import pickle
import argparse

from tqdm.auto import tqdm
from core.gen_models import (
	LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
from core.players import (
	PersuadeeModel, PersuaderModel, P4GSystemPlanner,
	PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.game import PersuasionGame
from core.helpers import DialogSession
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR
	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	if cmd_args.llm == 'code-davinci-002':
		backbone_model = OpenAIModel(cmd_args.llm)
		SysModel = PersuaderModel
		UsrModel = PersuadeeModel
		SysPlanner = P4GSystemPlanner
	elif cmd_args.llm in ['gpt-3.5-turbo']:
		backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'chatgpt':
		backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner

	game_ontology = PersuasionGame.get_game_ontology()
	sys_da = game_ontology['system']['dialog_acts']
	user_da = game_ontology['user']['dialog_acts']

	system = SysModel(
		sys_da,
		backbone_model, 
		conv_examples=[exp_1]
	)
	user = UsrModel(
		user_da,
		inference_args={
			"max_new_tokens": 128,
			"temperature": 1.1,
			"repetition_penalty": 1.0,
			"do_sample": True,
			"return_full_text": False,
		},
		backbone_model=backbone_model, 
		conv_examples=[exp_1]
	)
	planner = SysPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[exp_1]
	)
	game = PersuasionGame(system, user)

	with open("data/p4g/300_dialog_turn_based.pkl", "rb") as f:
		all_dialogs = pickle.load(f)

	num_dialogs = 20

	output = []  # for evaluation. [{did, context, ori_resp, new_resp, debug}, ...]
	# those dialogs has inappropriated content and will throw an error/be filtered with OPENAI models
	bad_dialogs = ['20180808-024552_152_live', '20180723-100140_767_live', '20180825-080802_964_live']
	num_done = 0
	pbar = tqdm(total=num_dialogs, desc="evaluating")
	for did in all_dialogs.keys():
		if did in bad_dialogs:
			print("skipping dialog id: ", did)
			continue
		if num_done == num_dialogs:
			break

		print("evaluating dialog id: ", did)
		context = ""
		no_error = True
		dialog = all_dialogs[did]
		
		state = game.init_dialog()
		for t, turn in enumerate(dialog["dialog"]):
			if len(turn["ee"]) == 0:  # ended
				break
			# also skip last turn as there is no evaluation
			if t == len(dialog["dialog"]) - 1:
				break

			usr_utt = " ".join(turn["ee"]).strip()
			usr_da = dialog["label"][t]["ee"][-1]

			# map to our dialog act
			if usr_da == "disagree-donation":
				usr_da = PersuasionGame.U_NoDonation
			elif usr_da == "negative-reaction-to-donation":
				usr_da = PersuasionGame.U_NegativeReaction
			elif usr_da == "positive-reaction-to-donation":
				usr_da = PersuasionGame.U_PositiveReaction
			elif usr_da == "agree-donation":
				usr_da = PersuasionGame.U_Donate
			else:
				usr_da = PersuasionGame.U_Neutral

			# game ended
			if usr_da == PersuasionGame.U_Donate:
				break

			# map sys as well
			sys_utt = " ".join(turn["er"]).strip()
			sys_da = set(dialog["label"][t]["er"])
			intersected_das = sys_da.intersection(system.dialog_acts)
			if len(intersected_das) == 0:
				sys_da = "other"
			else:
				sys_da = list(intersected_das)[-1]
			
			state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
			state.add_single(PersuasionGame.USR, usr_da, usr_utt)

			# update context for evaluation
			context = f"""
			{context}
			Persuader: {sys_utt}
			Persuadee: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			# mcts policy
			prior, v = planner.predict(state)
			greedy_policy = system.dialog_acts[np.argmax(prior)]
			try:
				next_best_state = game.get_next_state(state, np.argmax(prior))
			except Exception as e:
				bad_dialogs.append(did)
				no_error = False
				raise e
			greedy_pred_resp = next_best_state.history[-2][2]

			# next ground truth utterance
			human_resp = " ".join(dialog["dialog"][t + 1]["er"]).strip()
			next_sys_das = set(dialog["label"][t+1]["er"])
			next_intersected_das = next_sys_das.intersection(system.dialog_acts)
			if len(next_intersected_das) == 0:
				next_sys_da = "other"
			else:
				next_sys_da = list(next_intersected_das)[-1]

			# logging for debug
			debug_data = {
				"prior": prior,
				"da": greedy_policy,
				"v": v
			}

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': greedy_pred_resp,
				'new_da': greedy_policy,
				"debug": debug_data,
			}
			output.append(cmp_data)
		
		if no_error:
			with open(cmd_args.output, "wb") as f:
				pickle.dump(output, f)
			pbar.update(1)
			num_done += 1
	pbar.close()
	print(bad_dialogs)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--llm', type=str, default="code-davinci-002", choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt"], help='OpenAI model name')
	parser.add_argument('--gen_sentences', type=int, default=-1, help='max number of sentences to generate. -1 for no limit')
	parser.add_argument('--output', type=str, default="outputs/raw_prompt.pkl", help='output file')
	parser.parse_args()
	cmd_args = parser.parse_args()
	print("saving to", cmd_args.output)

	main(cmd_args)