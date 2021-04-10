import argparse
import json
import os
import numpy

parser = argparse.ArgumentParser()
def launch_params():
    parser.add_argument('--DATASET_LOC',
                        help='location of the dataset', 
                        default = '/home/huijiezhang/DeepReinforcementLearningMinecraft/EECS_545_Final_Project/data/MineRLTreechop-v0')
    parser.add_argument('--env',
                        help='the environment for minerl', 
                        default = 'MineRLTreechop-v0')


def create_actionspace(args):
    actionspace = {}
    if args.env == "MineRLTreechop-v0":
        ## the action space is shown in ./image/Action Discretization.png
        # actionspace = {
        #     0 : {'camera': [5, 0], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
        #     1 : {'camera': [-5, 0], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
        #     2 : {'camera': [0, 5], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
        #     3 : {'camera': [0, -5], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
        #     4 : {'camera': [0, 0], 'forward':1, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
        #     5 : {'camera': [0, 0], 'forward':1, 'left':0, 'right':0, 'back':0, 'jump':1, 'attack':1},
        #     6 : {'camera': [0, 0], 'forward':0, 'left':1, 'right':0, 'back':0, 'jump':0, 'attack':1},
        #     7 : {'camera': [0, 0], 'forward':0, 'left':0, 'right':1, 'back':0, 'jump':0, 'attack':1},
        #     8 : {'camera': [0, 0], 'forward':0, 'left':0, 'right':0, 'back':1, 'jump':0, 'attack':1},
        #     9 : {'camera': [0, 0], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':1, 'attack':1},
        # }
        actionspace = {
            0 : {'camera': [0, 5], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
            1 : {'camera': [0, -5], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
            2 : {'camera': [0, 0], 'forward':1, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
            3 : {'camera': [0, 0], 'forward':1, 'left':0, 'right':0, 'back':0, 'jump':1, 'attack':1},
            4 : {'camera': [0, 0], 'forward':0, 'left':1, 'right':0, 'back':0, 'jump':0, 'attack':1},
            5 : {'camera': [0, 0], 'forward':0, 'left':0, 'right':1, 'back':0, 'jump':0, 'attack':1},
            6 : {'camera': [0, 0], 'forward':0, 'left':0, 'right':0, 'back':1, 'jump':0, 'attack':1},
            7 : {'camera': [0, 0], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':1, 'attack':1},
        }
    assert len(actionspace.keys()) == args.actionNum, "action_num mismath"
    return actionspace


if __name__ == "__main__":
    launch_params()
    args = parser.parse_args()

    packages = os.listdir(args.DATASET_LOC)
    for package in packages:
        datapath = os.path.join(args.DATASET_LOC, package, "rendered.npz")
        videopath = os.path.join(args.DATASET_LOC, package, "recording.mp4")

    