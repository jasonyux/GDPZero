import argparse
import pickle
import logging
import os

from tqdm.auto import tqdm
from core.evaluator import P4GEvaluator
from core.gen_models import OpenAIModel, OpenAIChatModel, AzureOpenAIModel, AzureOpenAIChatModel


logger = logging.getLogger(__name__)


def main(args):
	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
		logger.setLevel(logging.DEBUG)
	
	# backbone_model = OpenAIModel('text-davinci-003')
	if args.judge in ['gpt-3.5-turbo']:
		backbone_model = OpenAIChatModel(args.judge)
	elif args.judge == 'chatgpt':
		backbone_model = AzureOpenAIChatModel(args.judge)
	else:
		raise ValueError(f"unknown judge: {args.judge}")
	evaluator = P4GEvaluator(backbone_model)

	with open(args.f, 'rb') as f:
		data: list = pickle.load(f)
	h2h_data = []
	if args.h2h:
		with open(args.h2h, 'rb') as f:
			h2h_data: list = pickle.load(f)
		assert(len(data) == len(h2h_data))
		assert(args.output != '')  # specify output path when doing h2h comparisons

	result = []
	stats = {
		'win': 0,  # if b=new_resp is better than a=ori_resp
		'draw': 0,
		'lose': 0,
	}
	for i, d in tqdm(enumerate(data[:]), total=len(data), desc="evaluating"):
		context = d['context']
		ori_resp = d['ori_resp']
		new_resp = d['new_resp']
		if len(h2h_data) > 0:
			ori_resp = h2h_data[i]['new_resp']
		
		winner, info = evaluator.evaluate(context, ori_resp, new_resp)

		# update winners
		if winner == 0:
			stats['lose'] += 1
		elif winner == 1:
			stats['win'] += 1
		else:
			stats['draw'] += 1
		
		info['winner'] = winner
		result.append(info)

	# save
	if args.output != '':
		output_file = args.output
	else:
		output_folder = os.path.join(os.path.dirname(args.f), 'evaluation')
		output_filename = os.path.basename(args.f).replace('.pkl', '_evaluated.pkl')
		output_file = os.path.join(output_folder, output_filename)
	with open(output_file, 'wb') as f:
		pickle.dump(result, f)

	# statistics
	win_rate = stats['win'] / sum(stats.values())
	print(f"win rate: {win_rate*100.0:.2f}%")
	print("stats: ", stats)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', type=str, help='path to the data file for comparing against human in p4g. See P4GEvaluator documentation to see the format of the file.')
	parser.add_argument('--judge', type=str, default='gpt-3.5-turbo', help='which judge to use.', choices=['gpt-3.5-turbo', 'chatgpt'])
	parser.add_argument('--h2h', type=str, default='', help='path to the data file for head to head comparison. If empty compare against human in p4g.')
	parser.add_argument("--output", type=str, default='', help="output file")
	parser.add_argument("--debug", action='store_true', help="debug mode")
	args = parser.parse_args()

	main(args)