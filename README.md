## Overview
<img src='image/finalDemo.gif' width="500"/>


This repo is the Final Project for EECS 545, University Michigan, Ann Arbor, Spring 2021 term. The project intends to train a tree chopping agent in the Minecraft environment using Deep Q-leanring from demonstration. The project is highly inspired from the competition MineRL, you could find more details about the competition [here](https://minerl.io/docs/).

## Table of contents
-----
  * [Requirements](#requirements)
  * [Dataset](#dataset)
  * [Train](#train)
  * [Evaluate](#evaluate)
------

## Requirements

The code depends on the following libraries:

* Python 3.7
* PyTorch 1.8.1
* CUDA 11.2
* MineRL
* PFRL

The envionment of Minecraft is wrapped by MineRL, so please follow the [document](https://minerl.io/docs/) to install the MineRL library first. Be sure you could successfully run the following test code from the document :

```python
import minerl
import gym
env = gym.make('MineRLNavigateDense-v0')


obs  = env.reset()
done = False
net_reward = 0

while not done:
    action = env.action_space.noop()

    action['camera'] = [0, 0.03*obs["compassAngle"]]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(
        action)

    net_reward += reward
    print("Total reward: ", net_reward)
```

PFRL in a library implementing some state-of-art deep reinforcement learning algorithm. Our project use it for the prioritized buffer. You could find more details about their work [here](https://pfrl.readthedocs.io/en/latest/index.html)

## Dataset

To download the demonstration data from human, you could follow the guidance in the [document](https://minerl.io/docs/) to the local path ``<YOUR LOCAL REPO PATH>/data/rawdata``, or more specific:

```bash
sudo gedit ~/.bashrc
```
Add ``export MINERL_DATA_ROOT=<YOUR LOCAL REPO PATH>/data/rawdata`` at the end of the file and save it, then:

```bash
source ~./bashrc
```

Then download the specific dataset ``MineRLTreechopVectorObf-v0``
```bash
python3 -m minerl.data.download "MineRLTreechopVectorObf-v0"
```

Then we should preprocess the dataset to extract the frames and calculate the actionspace:
```bash
python3 -u preprocess.py \
        --ROOT <YOUR LOCAL REPO PATH> \
        --DATASET_LOC <YOUR LOCAL REPO PATH>/data/rawdata/MineRLTreechopVectorObf-v0 \
        --actionNum 32 \
        --PREPARE_DATASET True \
        --n 25 \
```
It would generate the output frames in ``<YOUR LOCAL REPO PATH>/data/processdata`` and actionspace in ``<YOUR LOCAL REPO PATH>/actionspace``

## Train

If you like to train your own agent, be sure that your PC have at least 32 GB RAM:
```bash
python3 -u train.py \
        --ROOT <YOUR LOCAL REPO PATH> \
        --DATASET_LOC <YOUR LOCAL REPO PATH>/data/rawdata/MineRLTreechopVectorObf-v0 \
        --MODEL_SAVE <YOUR LOCAL REPO PATH>/saved_network\
        --actionNum 32 \
        --n 25 \
```

## Evaluate
We also provide our best train agent in ``<YOUR LOCAL REPO PATH>/saved_network/best_model.pt``, you could run it by:
```bash
python3 -u evaluate.py \
        --ROOT <YOUR LOCAL REPO PATH> \
        --DATASET_LOC <YOUR LOCAL REPO PATH>/data/rawdata/MineRLTreechopVectorObf-v0 \
        --MODEL_SAVE <YOUR LOCAL REPO PATH>/saved_network\
        --agentname best_model.pt
        --actionNum 32 \
        --n 25 \
```
The results of different architecture is shown in the table:




