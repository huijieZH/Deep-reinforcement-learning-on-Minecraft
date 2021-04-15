import argparse
import json
import os
import numpy as np
import imageio
import minerl
from sklearn.cluster import KMeans
import tqdm
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
def launch_params():
    parser.add_argument('--ROOT',
                        help='root',
                        default = '/home/huijiezhang/DeepReinforcementLearningMinecraft/EECS_545_Final_Project')
    parser.add_argument('--DATASET_LOC',
                        help='location of the dataset', 
                        default = '/home/huijiezhang/DeepReinforcementLearningMinecraft/EECS_545_Final_Project/data/rawdata/MineRLTreechop-v0')
    parser.add_argument('--env',
                        help='the environment for minerl', 
                        default = 'MineRLTreechop-v0')

    ##### actionspace
    parser.add_argument('--actionNum', type=int,
                        help='the number of discrete action combination',
                        default=32)
    parser.add_argument('--ACTIONSPACE_TYPE',choices=['manually', 'k_means'],
                        help='way to define the actionsapce',
                        default='k_means')
    
    ##### prepare dataset
    parser.add_argument('--PREPARE_DATASET',
                        help='if True, would automatically prepare dataset',
                        default=True)
    parser.add_argument('--DATA_TOTAL',
                        help='total data from demonstration',
                        default=400000)
    parser.add_argument('--DATA_PER_FILE',
                        help='total data from demonstration',
                        default=5000)

def create_actionspace(args):
    actionspace = {}
    actionspace_path = os.path.join(args.ROOT, "actionspace")
    if args.ACTIONSPACE_TYPE == 'manually':
        if args.env == "MineRLTreechop-v0":
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

    if args.ACTIONSPACE_TYPE == 'k_means':
        actionspaceFile = os.path.join(actionspace_path, args.env + "_" + args.ACTIONSPACE_TYPE + "_" + str(args.actionNum) + ".json")
        if os.path.exists(actionspaceFile):
            with open(actionspaceFile) as f:
                actionspace = json.load(f)
        else:
            data = minerl.data.make(args.env)
            act_vectors = []
            for _, act, _, _, _ in tqdm.tqdm(data.batch_iter(1, 1, 1)):
                act_vectors.append(vectorize(act))
                if len(act_vectors) >= 50000:
                    break
            # Reshape these the action batches
            acts = np.concatenate(act_vectors).reshape(-1, 10)

            # Use sklearn to cluster the demonstrated actions
            kmeans = KMeans(n_clusters=args.actionNum, random_state=0).fit(acts)
            for index in range(args.actionNum):
                actionspace[index] = {
                    'camera': [kmeans.cluster_centers_[index][0], kmeans.cluster_centers_[index][1]],
                    'attack': kmeans.cluster_centers_[index][2], 'back': kmeans.cluster_centers_[index][3],
                    'forward': kmeans.cluster_centers_[index][4], 'left': kmeans.cluster_centers_[index][5],
                    'right': kmeans.cluster_centers_[index][6], 'jump': kmeans.cluster_centers_[index][7],
                    'sprint': kmeans.cluster_centers_[index][8], 'sneak': kmeans.cluster_centers_[index][9],
                }
            with open(actionspaceFile, "w") as f:
                json.dump(actionspace, f, indent = True)
    return actionspace

def vectorize(act):
    vec1 = np.concatenate([act["camera"][0][0],act["attack"][0],act["back"][0]])
    vec2 = np.concatenate([act["forward"][0],act["left"][0],act["right"][0]])
    vec = np.concatenate([vec1,vec2,act["jump"][0]])
    return vec

def vectorize_v2(act):
    vec = np.array([ act["camera"][0],
                     act["camera"][1],
                     act["attack"],
                     act["back"],
                     act["forward"],
                     act["left"],
                     act["right"],
                     act["jump"], 
                     act['sprint'], 
                     act['sneak'] ])
    return vec

def prepare_dataset(args, actionspace):
    ## matrixlize the actionspace
    ## act_vec is actionNum X actionDim
    act_vec = []
    for index in actionspace:
        act = actionspace[index]
        act_vec.append(vectorize_v2(act))
    act_vec = np.stack(act_vec)

    root = os.path.join(args.ROOT, "data", "processdata" ,args.env + "_preprocess")
    if not os.path.exists(root):
        os.mkdir(root)

    total = 0
    per_file = 0
    fileindex = 0
    replaybuffer = []

    data = minerl.data.make(args.env)
    for current_state, action, reward, _, done in data.batch_iter(
                batch_size=1, num_epochs=1, seq_len=2):
        if total > args.DATA_TOTAL:
            break
        if per_file >= args.DATA_PER_FILE:
            np.save(os.path.join(root, "{:03d}".format(fileindex)), replaybuffer)
            per_file = 0
            fileindex += 1
            replaybuffer = []

        per_file += 1
        total += 1
        
        s = current_state['pov'][0, 0, :, :, :].astype(np.float32)
        s_normalize = (s - np.mean(s.reshape((-1, 3)), axis = 0))/np.std(s.reshape((-1, 3)), axis = 0)
        s = np.moveaxis(s_normalize, -1, 0)
        s_new = current_state['pov'][0, 1, :, :, :].astype(np.float32)
        s_new_normalize = (s_new - np.mean(s_new.reshape((-1, 3)), axis = 0))/np.std(s_new.reshape((-1, 3)), axis = 0)
        s_new = np.moveaxis(s_new_normalize, -1, 0)
        r = np.array([reward[0, 0]])
        t = np.array([not done[0, 0]])
        a = np.array([ action["camera"][0, 0, 0],
                     action["camera"][0, 1, 0],
                     action["attack"][0, 0],
                     action["back"][0, 0],
                     action["forward"][0, 0],
                     action["left"][0, 0],
                     action["right"][0, 0],
                     action["jump"][0, 0], 
                     action['sprint'][0, 0], 
                     action['sneak'][0, 0] ]).reshape((1, -1))
        action_index = np.argmin(np.linalg.norm(act_vec - a, axis = 1))
        action_one_hot = F.one_hot(torch.tensor([action_index]), args.actionNum).squeeze()
        replaybuffer.append((s.reshape((-1, 64, 64)), 
                            action_one_hot, 
                            r, 
                            s_new.reshape((-1, 64, 64)),
                            t,
                            ))
        

if __name__ == "__main__":
    launch_params()
    args = parser.parse_args()

    actionspace = create_actionspace(args)
    if args.PREPARE_DATASET:
        prepare_dataset(args, actionspace)

    