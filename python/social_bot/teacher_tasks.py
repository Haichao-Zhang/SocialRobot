# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A variety of teacher tasks.
"""

import math
import numpy as np
import os
import random

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo

from absl import logging
logging.set_verbosity(logging.DEBUG)


class GoalTask(teacher.Task):
    """
    A simple teacher task to find a goal.
    For this task, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.
    """
    def __init__(self,
                 max_steps=500,
                 goal_name="goal",
                 success_distance_thresh=0.5,
                 fail_distance_thresh=0.5,
                 random_range=1.0):
        """
        Args:
            max_steps (int): episode will end if not reaching gaol in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is given reward -1
            random_range (float): the goal's random position range
        """
        super().__init__()
        self._goal_name = goal_name
        self._success_distance_thresh = success_distance_thresh
        self._fail_distance_thresh = fail_distance_thresh
        self._max_steps = max_steps
        self._random_range = random_range
        self.task_vocab = ['hello', 'goal', 'well', 'done', 'failed', 'to']

    def run(self, agent, world):
        """
        Start a teaching episode for this task.
        Args:
            agent (pygazebo.Agent): the learning agent
            world (pygazebo.World): the simulation world
        """
        agent_sentence = yield
        # agent.reset() ## should reset to its initial loc #====>
        goal = world.get_agent(self._goal_name)
        loc, dir = agent.get_pose()  ## BUG: initial pose is always zero
        loc = np.array(loc)
        #self._move_goal(goal, loc)
        self._move_goal_relative(goal, (1, -1, 0), loc)
        steps_since_last_reward = 0
        while steps_since_last_reward < self._max_steps:
            steps_since_last_reward += 1
            loc, dir = agent.get_pose()
            goal_loc, _ = goal.get_pose()
            loc = np.array(loc)
            goal_loc = np.array(goal_loc)
            dist = np.linalg.norm(loc - goal_loc)
            if dist < self._success_distance_thresh:
                # dir from get_pose is (roll, pitch, roll)
                dir = np.array([math.cos(dir[2]), math.sin(dir[2])])
                goal_dir = (goal_loc[0:2] - loc[0:2]) / dist
                dot = sum(dir * goal_dir)
                if dot > 0.707:
                    # within 45 degrees of the agent direction
                    logging.debug("loc: " + str(loc) + " goal: " +
                                  str(goal_loc) + "dist: " + str(dist))
                    agent_sentence = yield TeacherAction(reward=1.0,
                                                         sentence="well done",
                                                         done=True)
                    steps_since_last_reward = 0
                    #self._move_goal(goal, loc)
                    self._move_goal_relative(goal, (1, -1, 0), loc)
                else:
                    agent_sentence = yield TeacherAction()
            elif dist > self._initial_dist + self._fail_distance_thresh:
                logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                              "dist: " + str(dist))
                yield TeacherAction(reward=-1.0, sentence="failed", done=True)
            else:
                agent_sentence = yield TeacherAction(sentence=self._goal_name)
        logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                      "dist: " + str(dist))
        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def _move_goal(self, goal, agent_loc):
        range = self._random_range
        while True:
            loc = (random.random() * range - range / 2,
                   random.random() * range - range / 2, 0)
            self._initial_dist = np.linalg.norm(loc - agent_loc)
            if self._initial_dist > self._success_distance_thresh:
                break
        goal.reset()
        #loc =（0， 0， 0）
        goal.set_pose((loc, (0, 0, 0)))

    def _move_goal_relative(self, goal, origin, agent_loc):
        """
        Move goal with respect to a specified position
        """
        range = self._random_range
        while True:
            loc = (random.random() * range - range / 2 + origin[0],
                   random.random() * range - range / 2 + origin[1],
                   random.random() * range + origin[2])
            self._initial_dist = np.linalg.norm(loc - agent_loc)
            if self._initial_dist > self._success_distance_thresh:
                break
        goal.reset()
        goal.set_pose((loc, (0, 0, 0)))

    def get_goal_name(self):
        """
        Args:
            None
        Returns:
            Goal's name at this episode
        """
        return self._goal_name

    def set_goal_name(self, goal_name):
        """
        Args:
            Goal's name
        Returns:
            None
        """
        logging.debug('Setting Goal to %s', goal_name)
        self._goal_name = goal_name


class IsoGoalTask(teacher.Task):
    """
    A simple teacher task to find a goal.
    It is an isotropic task, meaning only distance (no direction) is used as the metric for success.
    For this task, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.
    """
    def __init__(self,
                 fixed_agent_loc,
                 agent_name,
                 end_link_name,
                 goal_name="goal",
                 max_steps=500,
                 success_distance_thresh=0.2,
                 random_range=1.0):
        """
        Args:
            fixed_agent_loc: used for computing the relative goal location for policy learning
            end_link_name(string): the name of the link from the agent that will be used for determining succuess or failure
            max_steps (int): episode will end if not reaching gaol in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            random_range (float): the goal's random position range
        """
        super().__init__()
        self._fixed_agent_loc = fixed_agent_loc
        self._end_link_name = end_link_name
        self._goal_name = goal_name
        self._agent_name = agent_name
        self._success_distance_thresh = success_distance_thresh
        self._max_steps = max_steps
        self._random_range = random_range
        self.task_vocab = ['hello', 'goal', 'well', 'done', 'failed', 'to']

        self._goal_loc = None
        self._prev_dist = 1000

    def run(self, agent, world):
        """
        Start a teaching episode for this task.
        Args:
            agent (pygazebo.Agent): the learning agent
            world (pygazebo.World): the simulation world
        """
        def get_agent_end_loc():
            loc, dir = agent.get_link_pose(self._end_link_name)
            return loc

        agent_sentence = yield
        #agent.reset()  ## should reset to its initial loc #====>
        goal = world.get_agent(self._goal_name)
        # overwrite the input agent
        agent = world.get_agent(self._agent_name)  # expert

        # goal_loc, _ = goal.get_pose()
        # print(goal_loc)

        # if self._goal_loc is None:
        #     goal_loc, _ = goal.get_pose()
        #     goal_loc = np.array(goal_loc)
        #     self._goal_loc = goal_loc

        loc = self._fixed_agent_loc
        self._move_goal_relative(goal, loc, loc)
        steps_since_last_reward = 0
        while steps_since_last_reward < self._max_steps:
            steps_since_last_reward += 1

            end_loc = get_agent_end_loc()  # current end pos
            end_loc = np.array(end_loc)

            #goal_loc = self._goal_loc
            goal_loc, _ = goal.get_pose()
            # print("goal_loc=======")
            # print(goal_loc)
            # print("end_loc=======")
            # print(end_loc)
            dist = np.linalg.norm(end_loc - goal_loc)
            # relative coordinate
            relative_coord = np.array(goal_loc - self._fixed_agent_loc)
            relative_coord[-1] = 0
            goal_loc_str = str(relative_coord)

            if dist < self._success_distance_thresh:
                logging.debug("end_loc: " + str(end_loc) + " goal: " +
                              str(goal_loc) + "dist: " + str(dist))
                agent_sentence = yield TeacherAction(reward=1,
                                                     sentence=goal_loc_str,
                                                     done=True)
                steps_since_last_reward = 0
                agent.reset()  ## should reset to its initial loc #====>
                loc = self._fixed_agent_loc
                self._move_goal_relative(goal, loc, loc)
                self._goal_loc = None
                self._prev_dist = 1000
            else:
                logging.debug("_prevdist " + str(self._prev_dist) + "dist: " +
                              str(dist))
                if dist < self._prev_dist:
                    self._prev_dist = dist
                    # 1.5 ~ sqrt(2)
                    #shaping_reward = -0.01 + 0.1 * (1.5 - np.min([dist, 1.5]))
                shaping_reward = 0.1 * (self._prev_dist - dist)
                #print(shaping_reward)
                # else:
                #     shaping_reward = -0.1
                agent_sentence = yield TeacherAction(reward=shaping_reward,
                                                     sentence=goal_loc_str,
                                                     done=False)

        logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                      "dist: " + str(dist))
        # reset before the FAILURE end
        self._prev_dist = 1000
        yield TeacherAction(reward=-1.0, sentence=goal_loc_str, done=True)

    def _move_goal(self, goal, agent_loc):
        range = self._random_range
        while True:
            loc = (random.random() * range - range / 2,
                   random.random() * range - range / 2, 0)
            self._initial_dist = np.linalg.norm(loc - agent_loc)
            if self._initial_dist > self._success_distance_thresh:
                break
        goal.reset()
        #loc =（0， 0， 0）
        goal.set_pose((loc, (0, 0, 0)))

    def _move_goal_relative(self, goal, origin, agent_loc):
        """
        Move goal with respect to a specified position
        """
        range = self._random_range
        while True:
            loc = (random.random() * range - range / 2 + origin[0],
                   random.random() * range - range / 2 + origin[1], 0)

            # loc = (random.random() * range - range / 2 + origin[0],
            #        random.random() * range - range / 2 + origin[1],
            #        random.random() * range + origin[2])
            self._initial_dist = np.linalg.norm(loc - agent_loc)
            if self._initial_dist > self._success_distance_thresh:
                break
        goal.reset()
        goal.set_pose((loc, (0, 0, 0)))

    def get_goal_name(self):
        """
        Args:
            None
        Returns:
            Goal's name at this episode
        """
        return self._goal_name

    def set_goal_name(self, goal_name):
        """
        Args:
            Goal's name
        Returns:
            None
        """
        logging.debug('Setting Goal to %s', goal_name)
        self._goal_name = goal_name


class PoseGoalTask(teacher.Task):
    """
    A task for reaching a specified pose
    """
    def __init__(
            self,
            expert_name,
            expert_joints,
            agent_name,
            agent_joints,
            pose_func,
            max_steps=500,
            success_distance_thresh=0.2,
    ):
        """
        Args:
            target_pose_loc: the target pose location as the goal
            end_link_name(string): the name of the link from the agent that will be used for determining succuess or failure
            max_steps (int): episode will end if not reaching gaol in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            random_range (float): the goal's random position range
        """
        super().__init__()
        self._expert_name = expert_name
        self._expert_joints = expert_joints
        self._agent_name = agent_name
        self._agent_joints = agent_joints
        self._pose_func = pose_func
        self._success_distance_thresh = success_distance_thresh
        self._max_steps = max_steps
        self.task_vocab = ['hello', 'goal', 'well', 'done', 'failed', 'to']

        self._goal_loc = None
        self._prev_dist = 1000

    def run(self, agent, world):
        """
        Start a teaching episode for this task.
        Args:
            agent (pygazebo.Agent): the learning agent
            world (pygazebo.World): the simulation world
        """

        agent_sentence = yield
        agent.reset()  ## should reset to its initial loc #====>

        agent = world.get_agent(self._agent_name)  # test, goal
        expert = world.get_agent(self._expert_name)  # expert

        def get_pose(agent_name, joint_names):
            agent = world.get_agent(agent_name)  # test, goal
            return world._get_internal_states(agent, joint_names)

        def _get_internal_states(agent_name, agent_joints):
            agent = world.get_agent(agent_name)  # test, goal
            joint_pos = []
            joint_vel = []
            for joint_id in range(len(agent_joints)):
                joint_name = agent_joints[joint_id]
                joint_state = agent.get_joint_state(joint_name)
                joint_pos.append(joint_state.get_positions())
                joint_vel.append(joint_state.get_velocities())
            joint_pos = np.array(joint_pos).flatten()
            joint_vel = np.array(joint_vel).flatten()
            # pos of continous joint could be huge, wrap the range to [-pi, pi)
            joint_pos = (joint_pos + np.pi) % (2 * np.pi) - np.pi
            internal_states = np.concatenate((joint_pos, joint_vel), axis=0)
            return internal_states

        steps_since_last_reward = 0
        while steps_since_last_reward < self._max_steps:
            steps_since_last_reward += 1

            agent_pose = _get_internal_states(
                self._agent_name, self._agent_joints)  # current end pos
            expert_pose = _get_internal_states(self._expert_name,
                                               self._expert_joints)

            dist = np.linalg.norm(agent_pose - expert_pose)
            # relative coordinate
            # relative_coord = np.array(goal_loc - self._fixed_agent_loc)
            # relative_coord[-1] = 0
            goal_loc_str = str(expert_pose)

            if dist < self._success_distance_thresh:
                logging.debug("end_loc: " + str(agent_pose) + " goal: " +
                              str(expert_pose) + "dist: " + str(dist))
                agent_sentence = yield TeacherAction(reward=1,
                                                     sentence=goal_loc_str,
                                                     done=True)
                steps_since_last_reward = 0
                agent.reset()  ## should reset to its initial loc #====>
                loc = self.agent_pose
                #self._move_goal_relative(goal, loc, loc)
                #self._goal_loc = None
                self._prev_dist = 1000
            else:
                logging.debug("_prevdist " + str(self._prev_dist) + "dist: " +
                              str(dist))
                if dist < self._prev_dist:
                    self._prev_dist = dist
                    # 1.5 ~ sqrt(2)
                    #shaping_reward = -0.01 + 0.1 * (1.5 - np.min([dist, 1.5]))
                shaping_reward = 0.1 * (self._prev_dist - dist)
                #print(shaping_reward)
                # else:
                #     shaping_reward = -0.1
                agent_sentence = yield TeacherAction(reward=shaping_reward,
                                                     sentence=goal_loc_str,
                                                     done=False)

        logging.debug("loc: " + str(agent_pose) + " goal: " +
                      str(expert_pose) + "dist: " + str(dist))
        # reset before the FAILURE end
        self._prev_dist = 1000
        yield TeacherAction(reward=-1.0, sentence=goal_loc_str, done=True)
