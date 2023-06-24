import numpy as np
import logging
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
from core.mcts import MCTS, OpenLoopMCTS, OpenLoopMCTSParallel
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)


def play_gdpzero(backbone_model, args):
	args = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": args.num_mcts_sims,
		"max_realizations": args.max_realizations,
		"Q_0": args.Q_0,
	})

	game_ontology = PersuasionGame.get_game_ontology()
	sys_da = game_ontology['system']['dialog_acts']
	user_da = game_ontology['user']['dialog_acts']
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR

	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	system = PersuaderChatModel(
		sys_da,
		backbone_model, 
		conv_examples=[exp_1],
		inference_args={
			"temperature": 0.7,
			"do_sample": True,  # for MCTS open loop
			"return_full_text": False,
		}
	)
	user = PersuadeeChatModel(
		user_da,
		inference_args={
			"max_new_tokens": 128,
			"temperature": 1.1,
			"repetition_penalty": 1.0,
			"do_sample": True,  # for MCTS open loop
			"return_full_text": False,
		},
		backbone_model=backbone_model, 
		conv_examples=[exp_1]
	)
	planner = P4GChatSystemPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[exp_1]
	)

	game = PersuasionGame(system, user)
	state = game.init_dialog()

	# init
	state.add_single(game.SYS, 'greeting', "Hello. How are you?")
	print("You are now the Persuadee. Type 'q' to quit, and 'r' to restart.")
	print("Persuader: Hello. How are you?")

	your_utt = input("You: ")
	while your_utt.strip() != "q":
		if your_utt.strip() == "r":
			state = game.init_dialog()
			state.add_single(game.SYS, 'greeting', "Hello. How are you?")
			game.display(state)
			your_utt = input("You: ")
			continue
		
		# used for da prediction
		tmp_state = state.copy()
		tmp_state.add_single(game.USR, 'neutral', your_utt.strip())
		user_da = user.predict_da(tmp_state)

		logging.info(f"user_da: {user_da}")
		state.add_single(game.USR, user_da, your_utt.strip())

		# planning
		if isinstance(backbone_model, OpenAIModel):
			backbone_model._cached_generate.cache_clear()
		dialog_planner = OpenLoopMCTS(game, planner, args)
		for i in tqdm(range(args.num_MCTS_sims)):
			dialog_planner.search(state)

		mcts_policy = dialog_planner.get_action_prob(state)
		mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)]
		logger.info(f"mcts_policy: {mcts_policy}")
		logger.info(f"mcts_policy_next_da: {mcts_policy_next_da}")
		logger.info(dialog_planner.Q)

		sys_utt = dialog_planner.get_best_realization(state, np.argmax(mcts_policy))
		logging.info(f"sys_da: [{mcts_policy_next_da}]")
		print(f"Persuader: {sys_utt}")
		
		state.add_single(game.SYS, mcts_policy_next_da, sys_utt)
		your_utt = input("You: ")
	return


def play_raw_prompt(backbone_model):
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR
	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	game_ontology = PersuasionGame.get_game_ontology()
	sys_da = game_ontology['system']['dialog_acts']
	user_da = game_ontology['user']['dialog_acts']

	system = PersuaderChatModel(
		sys_da,
		backbone_model, 
		conv_examples=[exp_1]
	)
	user = PersuadeeChatModel(
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
	planner = P4GChatSystemPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[exp_1]
	)
	game = PersuasionGame(system, user)
	state = game.init_dialog()

	# init
	state.add_single(game.SYS, 'greeting', "Hello. How are you?")
	print("You are now the Persuadee. Type 'q' to quit, and 'r' to restart.")
	print("Persuader: Hello. How are you?")

	your_utt = input("You: ")
	while your_utt.strip() != "q":
		if your_utt.strip() == "r":
			state = game.init_dialog()
			state.add_single(game.SYS, 'greeting', "Hello. How are you?")
			game.display(state)
			your_utt = input("You: ")
			continue
		# used for da prediction
		state.add_single(game.USR, 'neutral', your_utt.strip())

		# planning
		prior, v = planner.predict(state)
		greedy_policy = system.dialog_acts[np.argmax(prior)]
		next_best_state = game.get_next_state(state, np.argmax(prior))
		greedy_pred_resp = next_best_state.history[-2][2]
		
		logging.info(f"sys_da: [{greedy_policy}]")
		print(f"Persuader: {greedy_pred_resp}")
		
		state.add_single(game.SYS, greedy_policy, greedy_pred_resp)
		your_utt = input("You: ")
	return


def main(args):
	if args.llm in ['code-davinci-002', 'text-davinci-003']:
		backbone_model = OpenAIModel(args.llm)
	elif args.llm in ['gpt-3.5-turbo']:
		backbone_model = OpenAIChatModel(args.llm, args.gen_sentences)
	elif args.llm == 'chatgpt':
		backbone_model = AzureOpenAIChatModel(args.llm, args.gen_sentences)

	if args.algo == 'gdpzero':
		print("using GDPZero as planning algorithm")
		play_gdpzero(backbone_model, args)
	elif args.algo == 'raw-prompt':
		print("using raw prompting as planning")
		play_raw_prompt(backbone_model)
	return


if __name__ == "__main__":
	# logging mode
	parser = argparse.ArgumentParser()
	parser.add_argument("--log", type=int, default=logging.WARNING, help="logging mode", choices=[logging.INFO, logging.DEBUG, logging.WARNING])
	parser.add_argument("--algo", type=str, default='gdpzero', choices=['gdpzero', 'raw-prompt'], help="planning algorithm")
	# used by PDP-Zero
	parser.add_argument('--llm', type=str, default="gpt-3.5-turbo", choices=["code-davinci-002", "gpt-3.5-turbo", "text-davinci-002", "chatgpt"], help='OpenAI model name')
	parser.add_argument('--gen_sentences', type=int, default=3, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
	parser.add_argument('--num_mcts_sims', type=int, default=10, help='number of mcts simulations')
	parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
	parser.add_argument('--Q_0', type=float, default=0.25, help='initial Q value for unitialized states. to control exploration')
	args = parser.parse_args()
	logging.basicConfig(level=args.log)
	logger.setLevel(args.log)

	main(args)