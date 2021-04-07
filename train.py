import minerl
import gym
import argparse
import torch
from torchsummary import summary
from network.DQN import DQN


parser = argparse.ArgumentParser()
def launch_params():
    ######################### about RL training #####################
    parser.add_argument('--env',
                        help='the environment for minerl to make', 
                        default = 'MineRLTreechop-v0')


    ######################### network architecture ##################

    ############# DQN 

    parser.add_argument('--actionNum', type = int,
                    help='the number of discrete action combination', 
                    default = 10)
    parser.add_argument('--device', 
                    help='running device for training model', 
                    default = 'cuda:0')



if __name__ == "__main__":
    launch_params()
    args = parser.parse_args()

    net = DQN(args).to(torch.device(args.device))
    summary(net, (12,64,64))
    # env = gym.make(args.env)
    # obs  = env.reset()
    # done = False
    # net_reward = 0

    # env.make_interactive(port=6666, realtime=True)

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
