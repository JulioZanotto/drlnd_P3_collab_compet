[//]: # (Image References)

[image1]: imgs/plots.png "Plotted Scores"

# Udacity Project Three: Collab-Compete (A Multi Agent Environment)
This project focus on training multi agent to be able to train to play together Tennis in an Unity environment.

## The Environment Description
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Learning Algorithm
To solve this environment, I have used the MADDPG (Multi Agent Deep Deterministic Policy Gradient) to train the agents. This variation of the DDPG Actor-Critic architecture, uses 2 decentralized Actors and 1 Centralized Critic, that means that the actor will only have acess to the scene and the critic will use the Q values of both actors to evaluate and update. So this uses the same off-policy with experience replay from the regular DDPG, as it was almost exactly the same from the previous project Continuous Control, but now the act step requires to send information for both agent from the scene and stores both returned actions, and so storing it the same way. The local and target architecture is maintained for training stability, and the network update process happens the same while sampling from the replay buffer for each agent and the critic network uses the actiors actions to evaluate. The rewards are store using the maximum between agents and therefore averaged to keep track of the algorithim evolution.


### The Actor-Critic Network Architecture 

    Actor(
        Input size of 24 (states)
        Fully Connected layer: input_size = 24, output_size = 128 (Activation ReLU)
        Batch Norm 1D: 128
        Fully Connected layer: input_size = 128, output_size = 128 (Activation ReLU)
        Batch Norm 1D: 128
        Fully Connected layer: input_size = 128, output_size = 2 (Output continuous values for moviment and hit)
    )

    Critic(
        Input size of 24 (states)
        Fully Connected layer: input_size = 24, output_size = 128 (Activation ReLU)
        Batch Norm 1D: 128
        Fully Connected layer: input_size = 128 + 2, output_size = 128 (Activation ReLU)
        Fully Connected layer: input_size = 128, output_size = 1 (Output the Q value)
    )

### hyper-parameter

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 128        # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR_ACTOR = 1e-3         # learning rate of the actor 
    LR_CRITIC = 1e-3        # learning rate of the critic
    WEIGHT_DECAY = 0.       # L2 weight decay

    The value of *GAMMA*, is responsible for the discounting the rewards for the episodes, so the nearest ones have more say in the final reward.
    

## Train The Network
    Episode 100	Average Score: 0.01	 Max score: 0.10
    Episode 200	Average Score: 0.02	 Max score: 0.10
    Episode 300	Average Score: 0.02	 Max score: 0.10
    Episode 400	Average Score: 0.02	 Max score: 0.20
    Episode 500	Average Score: 0.05	 Max score: 0.30
    Episode 600	Average Score: 0.04	 Max score: 0.10
    Episode 700	Average Score: 0.07	 Max score: 0.30
    Episode 800	Average Score: 0.11	 Max score: 0.40
    Episode 900	Average Score: 0.14	 Max score: 0.60
    Episode 1000	Average Score: 0.26	 Max score: 2.60
    Episode 1100	Average Score: 0.19	 Max score: 2.00
    Episode 1200	Average Score: 0.12	 Max score: 0.20
    **Environment solved in 1263 episodes, mean score: 0.50**
    Episode 1289	Average Score: 0.71

    Average score is the mean on the 100 last episodes, and I also plotted the max, just to understand the mean overall.
    Here we can see that in 1263 episodes the mean over the last 100 was 0.5, we can see throught the max that there we a lots of ups and downs, but leads in the end to the 0.5 mean.
    I chose to let the agent train a little more, and when it reached 0.7, to break the training loop, more than this the overall would be really instable, reaching up to 2.0 points to 0.2 in the mean score.

## Plotted Results

![Plotted Scores][image1]

## More Information
On [Tennis.ipynb](https://github.com/JulioZanotto/drlnd_P3_collab_compet/blob/main/Tennis.ipynb)


## Ideas for Future Work
The Agent had a very nice performance, and was able to solve the environment, but was a little instable after 1200 episodes, with some ups and downs.
Even thought, a few tweaks can be done, as change a little hyper parameters like the learning rate, this number was critical on the convergence, helping a lot on how fast the agent copuld learn. Also the OU noise parameters like the sigma and theta was critical for the convergence and any change would affect the exploration and thus the policy. I believe the most significant test would be also another method like D4PG or SAC (Soft Actor Critic).

It was shown in the last project and in a few others articles that PER (Prioritazed experience replay) had a significant improve in performance and algorithm estability.

I also want to try PPO to compare the results with DDPG, in a multi agent environment.

