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
import pickle

parser = argparse.ArgumentParser()
def launch_params():
    parser.add_argument('--ROOT',
                        help='root',
                        default = '/home/huijie/EECS545/EECS_545_Final_Project')
    parser.add_argument('--DATASET_LOC',
                        help='location of the dataset', 
                        default = '/home/huijie/EECS545/EECS_545_Final_Project/data/rawdata/MineRLTreechopVectorObf-v0')
    parser.add_argument('--env',
                        help='the environment for minerl', 
                        default = 'MineRLTreechopVectorObf-v0')

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
    # parser.add_argument('--DATA_TOTAL',
    #                     help='total data from demonstration',
    #                     default=400000)
    # parser.add_argument('--DATA_PER_FILE',
    #                     help='total data from demonstration',
    #                     default=5000)

def create_actionspace(args):
    actionspace = {}
    actionspace_path = os.path.join(args.ROOT, "actionspace")

    if args.ACTIONSPACE_TYPE == 'k_means':
        actionspaceFile = os.path.join(actionspace_path, args.env + "_" + args.ACTIONSPACE_TYPE + "_" + str(args.actionNum) + ".pickle")
        if os.path.exists(actionspaceFile):
            with open(actionspaceFile, 'rb') as f:
                kmeans = pickle.load(f)
        else:
            dat = minerl.data.make(args.env)
            act_vectors = []
            L = 100000
            for _, act, _, _,_ in tqdm.tqdm(dat.batch_iter(1, 1, 1, preload_buffer_size=20)):
                act_vectors.append(act['vector'])
                if len(act_vectors) > L:
                    break

            # Reshape these the action batches
            acts = np.concatenate(act_vectors).reshape(-1, 64)
            kmeans_acts = acts[:L]

            # Use sklearn to cluster the demonstrated actions
            kmeans = KMeans(n_clusters=args.actionNum, random_state=0).fit(kmeans_acts)
            print(kmeans)
            with open(actionspaceFile, "wb") as f:
                pickle.dump(kmeans, f)
    return kmeans

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

    root = os.path.join(args.ROOT, "data", "processdata" ,args.env + "_preprocess")
    if not os.path.exists(root):
        os.mkdir(root)

    videoindex = 0
    frame_index = 0
    video_root = os.path.join(root, "{:03d}".format(videoindex))
    if not os.path.exists(video_root):
        os.mkdir(video_root)
    
    test = True
    data = minerl.data.make(args.env)
    for current_state, action, reward, _, done in data.batch_iter(
                batch_size=1, num_epochs=1, seq_len=1):
        

        s = current_state['pov'][0, 0, :, :, :].astype(np.float32)/255.0
        s = np.moveaxis(s, -1, 0)
        if test:
            test = False
            print(s)
        r = np.array([reward[0, 0]])
        t = np.array([not done[0, 0]])
        action_index = actionspace.predict(action['vector'][0, :])
        action_one_hot = F.one_hot(torch.tensor([int(action_index)]), args.actionNum).squeeze()
        np.savez(os.path.join(video_root, "{:04d}.npz".format(frame_index)), s.reshape((-1, 64, 64)), action_one_hot, r, t)

        frame_index += 1
        
        if done:
            videoindex += 1
            frame_index = 0
            video_root = os.path.join(root, "{:03d}".format(videoindex))
            if not os.path.exists(video_root):
                os.mkdir(video_root)
        

if __name__ == "__main__":
    launch_params()
    args = parser.parse_args()

    actionspace = create_actionspace(args)
    if args.PREPARE_DATASET:
        prepare_dataset(args, actionspace)

    