# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import gym
import random
import social_bot
import logging
import time
import psutil
import os
import numpy as np


def main():
    # env = gym.make("SocialBot-Third-Person-v0")
    env = gym.make("SocialBot-Third-Person-Teacher-v0")
    steps = 0
    t0 = time.time()
    proc = psutil.Process(os.getpid())
    logging.info(" mem=%dM" % (proc.memory_info().rss // 1e6))
    for _ in range(10000000):
        obs = env.reset()
        action_space = env.action_space
        print(action_space)
        while True:
            # control = [(random.random() - 0.5) * 100
            #            for i in range(action_space.shape[0])]

            control = [(np.random.rand(_action_.shape[0]) - 0.5) * 100
                       for _action_ in action_space]
            print(control)
            obs, reward, done, info = env.step(control)
            steps += 1
            if done:
                logging.info("done reward: " + str(reward))
                break
        logging.info("steps=%s" % steps + " frame_rate=%s" %
                     (steps / (time.time() - t0)) + " mem=%dM" %
                     (proc.memory_info().rss // 1e6))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()