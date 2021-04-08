import minerl
import gym
import argparse

from network.DQN import DQN


parser = argparse.ArgumentParser()
def launch_params():
    ######################### about RL training #####################
    parser.add_argument('--env',
                        help='the environment for minerl to make', 
                        default = 'MineRLTreechop-v0')

    parser.add_argument('--gamma',  type = float,
                    help='parameters for DQN-Qnet architecture', 
                    default = 0.99)   
    parser.add_argument('--actionNum', type = int,
                    help='the number of discrete action combination', 
                    default = 10)

    ######################### network architecture ##################

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
                    default = 10)   
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
                    default = 50000)
    parser.add_argument('--CONTINUOUS_FRAME', type = float,
                    help='buffer size for replay', 
                    default = 4)

def create_actionspace(args):
    actionspace = {}
    if args.env == "MineRLTreechop-v0":
        ## the action space is shown in ./image/Action Discretization.png
        actionspace = {
            0 : {'camera': [5, 0], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
            1 : {'camera': [-5, 0], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
            2 : {'camera': [0, 5], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
            3 : {'camera': [0, -5], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
            4 : {'camera': [0, 0], 'forward':1, 'left':0, 'right':0, 'back':0, 'jump':0, 'attack':1},
            5 : {'camera': [0, 0], 'forward':1, 'left':0, 'right':0, 'back':0, 'jump':1, 'attack':1},
            6 : {'camera': [0, 0], 'forward':0, 'left':1, 'right':0, 'back':0, 'jump':0, 'attack':1},
            7 : {'camera': [0, 0], 'forward':0, 'left':0, 'right':1, 'back':0, 'jump':0, 'attack':1},
            8 : {'camera': [0, 0], 'forward':0, 'left':0, 'right':0, 'back':1, 'jump':0, 'attack':1},
            9 : {'camera': [0, 0], 'forward':0, 'left':0, 'right':0, 'back':0, 'jump':1, 'attack':1},
        }
    assert len(actionspace.keys()) == args.actionNum, "action_num mismath"
    return actionspace
    

if __name__ == "__main__":
    launch_params()
    args = parser.parse_args()

    actionspace = create_actionspace(args)

    # env = None
    env = gym.make(args.env)
    obs  = env.reset()
    net = DQN(args, actionspace, env)
    net.train()

    # while True:
    #     action = env.action_space.sample()

    #     # # action['camera'] = [0, 0.03*obs["compassAngle"]]
    #     # action['back'] = 0
    #     # action['forward'] = 1
    #     # action['jump'] = 1
    #     # action['attack'] = 1

    #     obs, reward, done, info = env.step(
    #         action)

    #     net_reward += reward
    #     print("Total reward: ", net_reward)
