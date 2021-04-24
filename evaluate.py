from train import launch_params
import minerl
import gym
import argparse

from network.DQN import DQN, DoubleDQN, DQFD

from prepocess import create_actionspace

parser = argparse.ArgumentParser()

def launch_params():
    ######################### prepocess ############################
    parser.add_argument('--ROOT',
                        help='root',
                        default = '/home/huijie/EECS545/EECS_545_Final_Project')
    parser.add_argument('--DATASET_LOC',
                        help='location of the dataset', 
                        default = '/home/huijie/EECS545/EECS_545_Final_Project/data/MineRLTreechopVectorObf-v0')
    parser.add_argument('--MODEL_SAVE',
                        help='location of the dataset', 
                        default = '/home/huijie/EECS545/EECS_545_Final_Project/saved_network/DQFD10step_marginloss_pretrain')
    parser.add_argument('--ACTIONSPACE_TYPE',choices=['manually', 'k_means'],
                        help='way to define the actionsapce',
                        default='k_means')
    parser.add_argument('--actionNum', type = int,
                    help='the number of discrete action combination', 
                    default = 32)
    ##### prepare dataset
    parser.add_argument('--PREPARE_DATASET',
                        help='if True, would automatically prepare dataset',
                        default=False)

    ######################### about RL training #####################
    parser.add_argument('--env',
                        help='the environment for minerl to make', 
                        default = 'MineRLTreechopVectorObf-v0')
    parser.add_argument('--port',
                        help='the port to launch Minecraft', 
                        default = 5656)

    parser.add_argument('--device', 
                    help='running device for training model', 
                    default = 'cuda:0')
    parser.add_argument('--dim_DQN_Qnet', type = int,
                    help='parameters for DQN-Qnet architecture', 
                    default = 32)   
    parser.add_argument('--CONTINUOUS_FRAME', type = int,
                    help='number of continuous frame to be stacked together', 
                    default = 1)  
    parser.add_argument('--mode',
                    help='mode should be train or evaluate', 
                    default = 'evaluate')  
    parser.add_argument('--agentname',
                    help='mode should be train or evaluate', 
                    default = 'DQFD_MineRLTreechopVectorObf-v0_0100000.pt')         
    parser.add_argument('--ACTION_UPDATE_INTERVAL', type = int,
                    help='step intervals between update action', 
                    default = 3)
    parser.add_argument('--EVALUATE_NUM', type = int,
                    help='step intervals between update action', 
                    default = 40)
    parser.add_argument('--EPSILON', type = float,
                    help='epsilon at the end of explore', 
                    default = 0.1)

if __name__ == "__main__":
    launch_params()
    args = parser.parse_args()

    ## create action space
    actionspace = create_actionspace(args)
    
    ## train network
    env = gym.make(args.env)

    env.make_interactive(port=args.port, realtime=True)

    obs  = env.reset()
    net = DQFD(args, actionspace, env)
    net.evaluate()