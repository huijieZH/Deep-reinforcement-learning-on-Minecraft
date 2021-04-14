import minerl
import gym

data = minerl.data.make('MineRLTreechop-v0')

for current_state, action, reward, next_state, done \
    in data.batch_iter(
        batch_size=1, num_epochs=1, seq_len=32):

        # Print the POV @ the first step of the sequence
        print(current_state['pov'][0])

        # Print the final reward pf the sequence!
        print(reward[-1])

        # Check if final (next_state) is terminal.
        print(done[-1])

        # ... do something with the data.
        print("At the end of trajectories the length"
              "can be < max_sequence_len", len(reward))

pass