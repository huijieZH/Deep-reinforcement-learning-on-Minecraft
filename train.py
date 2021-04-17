import minerl
import gym
import argparse

from network.DQN import DQN, DoubleDQN

from prepocess import create_actionspace, prepare_dataset

from dataloader.dataloader import MineCraftRLDataLoader


parser = argparse.ArgumentParser()
def launch_params():
    ######################### prepocess ############################
    parser.add_argument('--ROOT',
                        help='root',
                        default = '/home/huijiezhang/DeepReinforcementLearningMinecraft/EECS_545_Final_Project')
    parser.add_argument('--DATASET_LOC',
                        help='location of the dataset', 
                        default = '/home/huijiezhang/DeepReinforcementLearningMinecraft/EECS_545_Final_Project/data/MineRLTreechopVectorObf-v0')
    ####  actionspace
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
    parser.add_argument('--DATA_TOTAL',
                        help='total data from demonstration',
                        default=400000)
    parser.add_argument('--DATA_PER_FILE',
                        help='per file data from demonstration, please remove the last file if it has different number',
                        default=5000)

    ######################### about RL training #####################
    parser.add_argument('--env',
                        help='the environment for minerl to make', 
                        default = 'MineRLTreechopVectorObf-v0')
    parser.add_argument('--port',
                        help='the port to launch Minecraft', 
                        default = 5656)

    parser.add_argument('--gamma',  type = float,
                    help='parameters for DQN-Qnet architecture', 
                    default = 0.99)   

    parser.add_argument('--saveStep', type = int,
                    help='the number of step between savings', 
                    default = 5000)

    ######################### network architecture ##################
    parser.add_argument('--ARCH', choices=['DQN', 'DoubleDQN', 'DQFD'],
                    help='the architecture for reinforcement learning', 
                    default = 'DoubleDQN')

    ############# DQN 
    parser.add_argument('--LOADING_MODEL', 
                    help='if True, the network would automatically search saved network in ./saved_network \
                        ; if False, the network would train a new network', 
                    default = False)

    parser.add_argument('--device', 
                    help='running device for training model', 
                    default = 'cuda:0')
    parser.add_argument('--dim_DQN_Qnet', type = int,
                    help='parameters for DQN-Qnet architecture', 
                    default = 32)
    parser.add_argument('--OBSERVE', type = int,
                    help='step for observe', 
                    default = 200)   
    parser.add_argument('--EXPLORE', type = int,
                    help='step for explore, and after that the net would train', 
                    default = 600000)  
    parser.add_argument('--INITIAL_EPSILON', type = float,
                    help='epsilon at the beginning of explore', 
                    default = 0.5)
    parser.add_argument('--FINAL_EPSILON', type = float,
                    help='epsilon at the end of explore', 
                    default = 0.01)
    parser.add_argument('--REPLAY_MEMORY', type = float,
                    help='buffer size for replay', 
                    default = 10000)
    parser.add_argument('--CONTINUOUS_FRAME', type = int,
                    help='number of continuous frame to be stacked together', 
                    default = 1)
    parser.add_argument('--MINIBATCH', type = int,
                    help='mini batch size', 
                    default = 32)
    parser.add_argument('--UPDATE_INTERVAL', type = int,
                    help='update interval between current network and target network', 
                    default = 100)

    ######################### Dataset ##################
    parser.add_argument('--INITIAL_R', type = float,
                    help='initial ratio for the demonstration data in the training mini batch', 
                    default = 1.0)
    parser.add_argument('--FINAL_R', type = float,
                    help='final ratio for the demonstration data in the training mini batch', 
                    default = 0.1)
    
    #######################     PDDQN    #################
    parser.add_argument('--alpha', 
                    help='Exponent of errors to compute probabilities to sample', 
                    default = 0.6)
    parser.add_argument('--beta0',
                    help='Initial value of beta', 
                    default = 0.4)
    parser.add_argument('--betasteps',
                    help='Steps to anneal beta to 1', 
                    default = 2e5)
    parser.add_argument('--eps',
                    help='To revisit a step after its error becomes near zero', 
                    default = 0.01)
    parser.add_argument('--normalize_by_max',
                    help='Method to normalize weights', 
                    default = True)
    parser.add_argument('--error_min',
                    help='', 
                    default = 0)
    parser.add_argument('--error_max',
                    help='', 
                    default = 1)
    parser.add_argument('--num_steps',
                    help='', 
                    default = 1)                    
if __name__ == "__main__":
    launch_params()
    args = parser.parse_args()

    

    ## create action space
    actionspace = create_actionspace(args)

    # prepare dataset

    if args.PREPARE_DATASET:
        prepare_dataset(args, actionspace)
    
    ## train network
    env = gym.make(args.env)

    env.make_interactive(port=args.port, realtime=True)

    obs  = env.reset()
    net = DoubleDQN(args, actionspace, env)
    net.train()
