# GDP-Zero

This repository contains code for the EMNLP'2023 paper "[Prompt-Based Monte-Carlo Tree Search for Goal-Oriented Dialogue Policy Planning](https://arxiv.org/abs/2305.13660)".

## Prerequisites

1. **OPENAI API KEYS**: this project relies on prompting LLM to perform dialogue simulations
	```bash
	# for OpenAI users
	export OPENAI_API_KEY=sk-xxxx
	# for MS Azure users
	export MS_OPENAI_API_KEY=xxxx
	export MS_OPENAI_API_BASE="https://xxx.com"
	export MS_OPENAI_API_VERSION="xxx"
	export MS_OPENAI_API_CHAT_VERSION="xxx"
	```
2. Before executing any of the scripts, make sure to **add the project to the PYTHONPATH environment variable**:
	```bash
	> ~/GDPZero$ export PYTHONPATH=$(pwd)
	```

## Interactive Demo
You can converse with both PDP-Zero planning and raw-prompting based planning using the `interactive.py` script. **We note that its simulation speed is heavily dependent on OpenAI API's speed.**

The default option is to use PDP-Zero as the planning algorithm:
```bash
~/GDPZero$ python interactive.py
using GDPZero as planning algorithm
You are now the Persuadee. Type 'q' to quit, and 'r' to restart.
Persuader: Hello. How are you?
You: Hi, I am good. What about you?
100%|██████████████████| 10/10 [00:32<00:00, 3.17s/it]
Persuader: I'm doing well, thank you. I was just wondering if you've heard of the charity called Save the Children?
You: No I have not. What does this charity do?
100%|██████████████████| 10/10 [00:37<00:00, 3.69s/it]
Persuader: Save the Children is an organization that helps children in developing countries by providing relief and promoting children's rights. It's a great charity that makes a positive impact on so many children's lives. They help with things like education, health care, and safety.
You: 
```
in the above example, PDP-Zero performs a tree search with `n=10` simulations and `k=3` realizations per state. You can change these parameters using the `--num_mcts_sims` and `--max_realizations` flags, respectively. See `interactive.py -h` and the [Experiments](#experiments) section for more details.
```bash
~/GDPZero$ python interactive.py -h
optional arguments:
  -h, --help            show this help message and exit
  --log {20,10,30}      logging mode
  --algo {gdpzero,raw-prompt}
                        planning algorithm
  --llm {code-davinci-002,gpt-3.5-turbo,text-davinci-002,chatgpt}
                        OpenAI model name
  --gen_sentences GEN_SENTENCES
                        number of sentences to generate from the llm. Longer ones will be truncated by nltk.
  --num_mcts_sims NUM_MCTS_SIMS
                        number of mcts simulations
  --max_realizations MAX_REALIZATIONS
                        number of realizations per mcts state
  --Q_0 Q_0             initial Q value for unitialized states. to control exploration
```

## Experiments

We mainly test PDP-Zero on the [PersuasionForGood](https://arxiv.org/abs/1906.06725) dataset. The scripts below will take the first 20 dialogues from the dataset, and perform planning/response generation for each turn. The output is a pickle file containing the generated responses and the corresponding contexts. This output pickle file is then used for evaluation (see [Static Evaluation](#static-evaluation)).

*PDP-Zero*:
```bash
> ~/GDPZero$ python runners/gdpzero.py -h
optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       output file
  --llm {code-davinci-002,chatgpt,gpt-3.5-turbo}
                        OpenAI model name
  --gen_sentences GEN_SENTENCES
                        number of sentences to generate from the llm. Longer ones will be truncated by nltk.
  --num_mcts_sims NUM_MCTS_SIMS
                        number of mcts simulations
  --max_realizations MAX_REALIZATIONS
                        number of realizations per mcts state
  --Q_0 Q_0             initial Q value for unitialized states. to control exploration
  --num_dialogs NUM_DIALOGS
                        number of dialogs to test MCTS on
  --debug               debug mode
```
for example, using `gpt-3.5-turbo` as backbone with `n=10` simulations, `k=3` realizations per state, and `Q_0=0.25` for exploration, do:
```bash
> python runners/gdpzero.py --output outputs/gdpzero.pkl --llm gpt-3.5-turbo --num_mcts_sims 10 --max_realizations 3 --Q_0 0.25
```

*Baseline*:
```bash
> ~/GDPZero$ python runners/raw_prompting.py -h
optional arguments:
  -h, --help            show this help message and exit
  --llm {code-davinci-002,gpt-3.5-turbo,chatgpt}
                        OpenAI model name
  --gen_sentences GEN_SENTENCES
                        max number of sentences to generate. -1 for no limit
  --output OUTPUT       output file
```
for example, using `gpt-3.5-turbo` as backbone, do
```bash
> python runners/raw_prompting.py --output outputs/chatgpt_raw_prompt.pkl --llm gpt-3.5-turbo
```

*Ablations*: 
```bash
# without OpenLoop
~/GDPZero$ python runners/gdpzero_noopenloop.py -h
optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       output file
  --llm {code-davinci-002,gpt-3.5-turbo,chatgpt}
                        OpenAI model name
  --gen_sentences GEN_SENTENCES
                        max number of sentences to generate
  --num_mcts_sims NUM_MCTS_SIMS
                        number of mcts simulations
```
```bash
# without response selection
~/GDPZero$ python runners/gdpzero_noRS.py -h
optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       output file
  --llm {code-davinci-002,gpt-3.5-turbo,chatgpt}
                        OpenAI model name
  --gen_sentences GEN_SENTENCES
                        max number of sentences to generate
  --num_mcts_sims NUM_MCTS_SIMS
                        number of mcts simulations
  --max_realizations MAX_REALIZATIONS
                        number of realizations per mcts state
  --Q_0 Q_0             initial Q value for unitialized states. to control exploration
```
where most of the arguments are the same ones in `gdpzero.py`.

## Static Evaluation
We mainly use `gpt-3.5-turbo` as the judge for static evaluation. To evaluate the planned dialogues from the [Experiments](Experiments) section, use the `test.py` script which prompts ChatGPT to compare the responses between either human demonstrations in P4G or against some generated responses:
```bash
> ~/GDPZero$ python test.py -h
optional arguments:
  -h, --help            show this help message and exit
  -f F                  path to the data file for comparing against human in p4g. See P4GEvaluator documentation to see the format of the file.
  --judge {gpt-3.5-turbo,chatgpt}
                        which judge to use.
  --h2h H2H             path to the data file for head to head comparison. If empty compare against human in p4g.
  --output OUTPUT       output file
  --debug               debug mode
```
For example to compare `outputs/gdpzero_50sims_3rlz_0.25Q0_20dialogs.pkl`
- against human demonstration
	```bash
	> ~/GDPZero$ python test.py -f outputs/gdpzero_50sims_3rlz_20dialogs.pkl --output eval.pkl --judge gpt-3.5-turbo
	evaluating: 100%|███████████████| 154/154 [03:49<00:00,  1.49s/it]
	win rate: 93.51%
	stats:  {'win': 144, 'draw': 0, 'lose': 10}
	```
- head-to-head comparison against ChatGPT generated responses (e.g. `outputs/chatgpt_raw_prompt.pkl`, see [Experiments](#experiments) section for more details)
	```bash
	> ~/GDPZero$ python test.py -f outputs/gdpzero_50sims_3rlz_20dialogs.pkl --h2h outputs/chatgpt_raw_prompt.pkl --output eval.pkl --judge gpt-3.5-turbo
	evaluating: 100%|███████████████| 154/154 [03:29<00:00,  1.36s/it]
	win rate: 59.09%
	stats:  {'win': 91, 'draw': 2, 'lose': 61}
	```

## Examples

We provided some example generations in the `output` directory. For instance:
```bash
output
├── chatgpt_raw_prompt.pkl  # chatgpt baseline
├── gdpzero_10sims_3rlz_0.25Q0_20dialogs.pkl  # gdp-zero with n=10, k=3, Q_0=0.25
├── gdpzero_10sims_v_chatgpt.pkl  # evaluation result of gdp-zero against chatgpt
├── gdpzero_20sims_3rlz_0.0Q0_20dialogs.pkl
├── gdpzero_50sims_3rlz_0.0Q0_20dialogs.pkl
└── gdpzero_5sims_3rlz_0.0Q0_20dialogs.pkl
```