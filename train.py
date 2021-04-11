import minerl
import gym
import argparse

from network.DQN import DQN, DoubleDQN

from prepocess import create_actionspace

parser = argparse.ArgumentParser()
def launch_params():
    ######################### about RL training #####################
    parser.add_argument('--env',
                        help='the environment for minerl to make', 
                        default = 'MineRLTreechop-v0')
    parser.add_argument('--port',
                        help='the port to launch Minecraft', 
                        default = 5656)

    parser.add_argument('--gamma',  type = float,
                    help='parameters for DQN-Qnet architecture', 
                    default = 0.99)   
    parser.add_argument('--actionNum', type = int,
                    help='the number of discrete action combination', 
                    default = 9)
    parser.add_argument('--saveStep', type = int,
                    help='the number of step between savings', 
                    default = 50000)

    ######################### network architecture ##################

    ############# DQN 
    parser.add_argument('--LOADING_MODEL', 
                    help='if True, the network would automatically search saved network in ./saved_network \
                        ; if False, the network would train a new network', 
                    default = False)

    parser.add_argument('--device', 
                    help='running device for training model', 
                    default = 'cpu')
    parser.add_argument('--dim_DQN_Qnet', type = int,
                    help='parameters for DQN-Qnet architecture', 
                    default = 32)
    parser.add_argument('--OBSERVE', type = int,
                    help='step for observe', 
                    default = 20000)   
    parser.add_argument('--EXPLORE', type = int,
                    help='step for explore, and after that the net would train', 
                    default = 300000)  
    parser.add_argument('--INITIAL_EPSILON', type = float,
                    help='epsilon at the beginning of explore', 
                    default = 0.1)
    parser.add_argument('--FINAL_EPSILON', type = float,
                    help='epsilon at the end of explore', 
                    default = 0.0001)
    parser.add_argument('--REPLAY_MEMORY', type = float,
                    help='buffer size for replay', 
                    default = 10000)
    parser.add_argument('--CONTINUOUS_FRAME', type = int,
                    help='number of continuous frame to be stacked together', 
                    default = 4)
    parser.add_argument('--MINIBATCH', type = int,
                    help='mini batch size', 
                    default = 32)
    parser.add_argument('--UPDATE_INTERVAL', type = int,
                    help='update interval between current network and target network', 
                    default = 10)


    

if __name__ == "__main__":
    launch_params()
    args = parser.parse_args()

    actionspace = create_actionspace(args)

    # env = None
    env = gym.make(args.env)

    env.make_interactive(port=args.port, realtime=True)

    obs  = env.reset()
    net = DoubleDQN(args, actionspace, env)
    net.train()
