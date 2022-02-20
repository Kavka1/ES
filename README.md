# Evolution Strategies

Framework of the Natural Evolutionary Strategies (OpenAI version).

OpenAI paper link: https://arxiv.org/abs/1703.03864

Other reproductions of the Evolutionary Strategies Algorithms might be added later, including:
-  CMA-ES(https://arxiv.org/abs/1604.00772)
-  Guided ES(http://proceedings.mlr.press/v97/maheswaranathan19a.html)
-  Structure NES(http://proceedings.mlr.press/v80/choromanski18a.html) 

## Dependencies
Install dependencies with `pip install -r requirements.txt`

Issues about `mujoco_py`, please refer to https://github.com/openai/mujoco-py.

## Train
- Change the `result_path` in the `config.yaml` to the path you want to save the training results.
- Change the path on line 13 of the `train.py` script.
- Run with `python train.py`

## Evaluate with demos
- If using the existing results, please change the `result_path` in the `config.yaml` to the corresponding experiment result path. (If use the results you have trained, skip this step.)
- Run with `python demo.py --exp_result [the results path without the last '/'] --num_rollout [num of episodes]`

