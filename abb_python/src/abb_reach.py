#! /usr/bin/env python

import ray
import time
import rospy
import cv2
import tf
import os
import atexit
import json
import random
import signal
import subprocess
import sys
import traceback
import numpy as np
import random
from geometry_msgs.msg import PointStamped
import image_geometry
from gym import utils
from gym import spaces
from sensor_msgs.msg import JointState
sys.path.insert(0, '/home/arl/env/src')
import abb_rob_env


def launch_gazebo():
    ros_port = random.randint(10000, 14000)
    gazebo_port = ros_port + 1
    print("Initializing new gazebo instance...")
    # Create a new server process and start the client.
    gazebo_process = subprocess.Popen(['/home/arl/env/src/abb_irb120_gazebo/gazeboSimulation0.sh', str(gazebo_port), str(ros_port)],
                                       preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
    gazebo_pgid = os.getpgid(gazebo_process.pid)
    print("Launched Gazebo Process with PGID " + str(gazebo_pgid))

    return ros_port, gazebo_port, gazebo_pgid

def launch_moveit(ros_port, gazebo_port):
    print("Initializing new moveit instance...")
    moveit_process = subprocess.Popen(
        ['/home/arl/env/src/abb_irb120_gazebo/moveitSimulation0.sh', str(gazebo_port), str(ros_port)],
        preexec_fn=os.setsid, stdout=open(os.devnull, "w"))

    moveit_pgid = os.getpgid(moveit_process.pid)
    print("Launched MoveIt Process with PGID " + str(moveit_pgid))

    return moveit_pgid


class ABBReachEnv(abb_rob_env.Abbenv, utils.EzPickle):
    def __init__(self, env_config):
        
        #print("Entered Reach Env")

        #The last element of the pose encodes the sine of the angle between 90 and -90 degrees
        #need to check limits and normalize it+ sin encoding of rot limit
        XYZlim = 0.8   # limits for states, don't use for goals
        X_max = 1.1    # limit for both goals in x
        Ylim = 0.50    # limit for both goals in y
        Zlim = 1.0     # upper limit for both goals in z
        NZlim = -0.1   # lower limit for both goals in z
        self.observation_space = spaces.Dict({'achieved': spaces.Box(low=np.tile(np.array([-0.3, -Ylim, NZlim, -0.1, -0.1]),(5,1)),
                                                                       high=np.tile(np.array([X_max, Ylim, Zlim, 1.1, 1.1]),(5,1)), dtype=np.float32),
                                               'desired': spaces.Box(low=np.array([-0.3, -Ylim, NZlim, -0.1, -0.1]),
                                                                       high=np.array([X_max, Ylim, Zlim, 1.1, 1.1]), dtype=np.float32),
                                               'image': spaces.Box(low=0, high=1.1, shape=(234, 234, 4), dtype=np.float32),
                                               'states': spaces.Box(low=np.array([-XYZlim, -XYZlim, 0, -1.1, -0.1]),
                                                                    high=np.array([XYZlim, XYZlim, XYZlim, 1.1, 1.1]), dtype=np.float32),
                                            })

        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1]),
                                        high=np.array([1, 1, 1, 1, 1]), dtype=np.float32)        

        if env_config["worker_number"] > 0:
            self.index = env_config["worker_number"]
            self.ros_port, self.gazebo_port, self.gazebo_pgid = launch_gazebo()

            file = open("gazebo_process_" + str(self.index) + ".txt", "w")
            file.write(str(self.gazebo_pgid))
            file.close()

            file = open("ros_port_" + str(self.index) + ".txt", "w")
            file.write(str(self.ros_port))
            file.close()
            #print('PORT NUMBER',self.ros_port)

            os.environ['ROS_MASTER_URI'] = "http://localhost:" + str(self.ros_port) + '/'
            os.environ['GAZEBO_MASTER_URI'] = "http://localhost:" + str(self.gazebo_port) + '/'
            rospy.wait_for_service('/gazebo/get_world_properties')

            self.moveit_pgid = launch_moveit(self.ros_port, self.gazebo_port)
            file = open("moveit_process_" + str(self.index) + ".txt", "w")
            file.write(str(self.moveit_pgid))
            file.close()

            time.sleep(5)
 
            #print("Entered Reach Env")
            rospy.init_node('env', anonymous=True)

            self.cam_model = image_geometry.PinholeCameraModel()

            self.get_params()

            # intializing robot env
            abb_rob_env.Abbenv.__init__(self)
            utils.EzPickle.__init__(self)

            # calling setup env.
            #print("Call env setup")
            self.pose = np.array([0, 0, 0])
            self._env_setup(initial_qpos=self.init_pos)
        
        #print("Exit Reach Env")

    def get_params(self):
        # get configuration parameters
        """
        self.n_actions = rospy.get_param('/fetch/n_actions')
        self.has_object = rospy.get_param('/fetch/has_object')
        self.block_gripper = rospy.get_param('/fetch/block_gripper')
        self.n_substeps = rospy.get_param('/fetch/n_substeps')
        self.gripper_extra_height = rospy.get_param('/fetch/gripper_extra_height')
        self.target_in_the_air = rospy.get_param('/fetch/target_in_the_air')
        self.target_offset = rospy.get_param('/fetch/target_offset')
        self.obj_range = rospy.get_param('/fetch/obj_range')
        self.target_range = rospy.get_param('/fetch/target_range')
        self.distance_threshold = rospy.get_param('/fetch/distance_threshold')
        self.init_pos = rospy.get_param('/fetch/init_pos')
        self.reward_type = rospy.get_param('/fetch/reward_type')
        """
        #from has_object to target range not used except target in the air
        self.n_actions = 5
        self.has_object = True
        self.block_gripper = False
        self.n_substeps = 20
        self.gripper_extra_height = 0.2
        self.target_in_the_air = True
        self.target_offset = 0.0
        self.obj_range = 0.5
        self.target_range = 0.15
        self.distance_threshold = 0.01
        self.reward_type = "sparse"
        self.init_pos = {
            'joint0': 0.0,
            'joint1': 0.0,
            'joint2': 0.0,
            'joint3': 0.0,
            'joint4': 0.0,
            'joint5': 0.0
        }

    def _set_action(self, action):

        # Take action
        assert action.shape == (self.n_actions,)
        action_w_gripper_state = action.copy()  # ensure that we don't change the action outside of this scope
        self.pose += np.array(action_w_gripper_state[0:3]) * 0.05
        #print(self.pose)
        self.pose = np.around(self.pose, decimals=5)
        action_w_gripper_state = np.around(action_w_gripper_state, decimals=5)
        self.set_trajectory_ee([self.pose[0], self.pose[1], self.pose[2], np.arcsin(action_w_gripper_state[3]), action_w_gripper_state[4]])

    def _get_obs(self):

        ###################################################################################################
        #getting the image for the current observation the image should be a numpy array encoding a RGB-D stack

        image = self.get_stacked_image()

        ###################################################################################################
        # The pose of the end effector consists of the end effector 3D translation and rotation about the Z axis
        # in addition to an indication of the aperature of the gripper and command success

        grip_pose, grip_state = self.get_ee_pose()
        grip_pos_array = np.array([grip_pose.pose.position.x, grip_pose.pose.position.y, grip_pose.pose.position.z])

        grip_rpy = self.get_ee_rpy()
        grip_rot_array = np.array([np.sin(grip_rpy.y)])

        self.pose = np.array([grip_pose.pose.position.x, grip_pose.pose.position.y, grip_pose.pose.position.z])

        #Check whether to add success or if the gripper is opened or closed only
        self.gripper_success_only = False

        if self.gripper_success_only:
            gripper_state = np.array([grip_state[1]]) #is task reached? bnghyar el reach emta?
        else:
            gripper_state = np.array([grip_state[0]]) #is gripper open?


        obs = np.concatenate([ grip_pos_array, grip_rot_array, gripper_state])

        #Get the object poses from the simulator and sample achieved goals
        object_pos = self.get_model_states()
        achieved_goal = self._sample_achieved_goal(object_pos)
        
        return {
            'image': image/255,
            'states': obs.copy(),
            'desired': self.goal.copy(),
            'achieved': achieved_goal.copy(),  
              }

    def _is_done(self, observations):
        #print("ana f distance l is done")
        d = self.goal_distance(observations['achieved'], self.goal) #hwa msh kda hena kul showia hy3od y sample random object msh hyb2a el desired?

        if d < self.distance_threshold:
            done = (d < self.distance_threshold).astype(np.float32)
        else:
            done = (d < self.distance_threshold).astype(np.float32)

        return bool(done)

    def _compute_reward(self, observations, done): #3mora by2olak tamam
        
        #print("ana f distance l reward")
        d = self.goal_distance(observations['achieved'], self.goal)
        #if self.reward_type == 'sparse':
        #    #print(d < self.distance_threshold)
        #    if d < self.distance_threshold:
        #        return (d < self.distance_threshold).astype(np.float32)
        #    else:
        #        #print(-0.05 * ((d > self.distance_threshold).astype(np.float32)))
        #        return -0.05 * ((d > self.distance_threshold).astype(np.float32))
        #else:
        #    assert False
        #    return -d
        if self.reward_type == 'sparse':
            c = np.array(d < self.distance_threshold) #can be squeezed if needed check
            c = np.where(c == False, -0.05, 1)
            #print("reward = "+str(c))
            return float(c)
        else:
            #print("reward = "+str(-d))
            return -d


    def compute_reward(self, achieved_goal, goal, info):
        #print("ana f distance l rewardzzzz")
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            c = np.array(d < self.distance_threshold) #can be squeezed if needed check
            c = np.where(c == False, -0.05, 1)
            return c
        else:
            return -d

    def _set_init_pose(self):
        """Sets the Robot in its init pose"""
        self.gazebo.unpauseSim()
        gripper_target = np.array([0.498, 0.005, 0.431])
        gripper_rotation = np.array([0, 0, 0])
        action = np.concatenate([gripper_target, gripper_rotation], None)
        self.pose = np.array([0, 0, 0])
        self.set_trajectory_ee(action)

        return True

    def goal_distance(self, goal_a, goal_b): #3mora l3b hna
        if len(goal_a.shape) == 3:
            if len(goal_a.shape) > len(goal_b.shape):
                goal_a = np.swapaxes(goal_a, 0, 1)
                return np.around(np.linalg.norm(goal_a - goal_b, axis=len(goal_a.shape) - 1), decimals=3)

            else:
                goal_b = np.swapaxes(goal_b, 0, 1)
                return np.around(np.linalg.norm(goal_a - goal_b, axis=len(goal_b.shape) - 1), decimals=3)
        else:

            if len(goal_a.shape) > len(goal_b.shape):

                i = np.where(goal_a[:,3:] == goal_b[3:])
              #  i = np.asarray(i, dtype=np.int64)  3mora kan byl3b
            # add np around to all
                #print(np.around(np.linalg.norm(goal_a[i[0][0]] - goal_b), decimals=3))
                return np.around(np.linalg.norm(goal_a[i[0][0]] - goal_b), decimals=3)

            else:

                i = np.where(goal_b[:,3:] == goal_a[3:])
               # i = np.asarray(i, dtype=np.int64)  3mora kan byl3b

            # add np around to all
                #print(np.around(np.linalg.norm(goal_a - goal_b[i[0][0]]), decimals=3))
                return np.around(np.linalg.norm(goal_a - goal_b[i[0][0]]), decimals=3)

    def _sample_goal(self):
        #this function should be modified to be one of the objects pose in addition to any 3d position in the
        # vicinity with placing position 50% of the times in air

        print("sample goal")
        #get the xyz position of a random object in the world
        _, position = random.choice(list(self.init_model_states.items()))

        goal_position = position + np.random.uniform(-self.obj_range, self.obj_range, size=3)
        goal_position[0] = np.clip(goal_position[0], 0, 1.09)
        goal_position[1] = np.clip(goal_position[1], -0.49, 0.49)
        goal_position[0] = np.abs(goal_position[0])
        goal_position[2] = position[2]
        #get the pixel location of the object in the image
        pixel_position = self._return_pixel(position[0], position[1], position[2])
        goal = np.concatenate([goal_position, pixel_position])
        if self.target_in_the_air and np.random.uniform() < 0.5:
           goal[2] += np.random.uniform(0, 0.45)
        #goal[:3]=np.around(goal[:3],decimals=3)
        print('goal = ' + str(goal))
        return goal

    def _sample_achieved_goal(self, object_pos): #3mora l3b hna

        # this should sample the changed position of any object
        achieved_goal = []
        current_pos = object_pos.copy()
        #changed_models = []
        """for model, position in current_pos.items(): #wezza kan 3yz yekhaly el achieved goal array fe kul el objects ele et7rakt then ye sample hwa brahto
            #print(position, self.init_model_states[model])
            if np.linalg.norm(position[:]-self.init_model_states[model]) > 0.05:
               #print( np.linalg.norm(position[:]-self.init_model_states[model]))
               changed_models.append(model)
            #print(changed_models)"""

        #if len(changed_models) != 0:
            #random_model = random.choice(changed_models)
            #print(self.init_model_states[random_model])
            #print(achieved_goal[random_model])

        for model, position in current_pos.items():
            #if model in changed_models:
            initial = self._return_pixel(self.init_model_states[model][0], self.init_model_states[model][1], self.init_model_states[model][2])
            achieved_goal.append(np.concatenate([current_pos[model], initial]))
                #else:
                #    achieved_goal.append(np.zeros((5,)))
        #else:
         #   achieved_goal = np.zeros((5,5,)) ##3mora l3b hna
        #print('achgoal' + str(np.array(achieved_goal)))
        return np.array(achieved_goal)

    def _return_pixel(self, x, y, z):
        #Takes the xyz position, transforms it to the camera frame, then returns it's pixel position in the image
        self.world_point.point.x = x
        self.world_point.point.y = y
        self.world_point.point.z = z
        self.listener.waitForTransform("camera", "world", rospy.Time(), rospy.Duration(5.0))
        self.world_point.header.stamp = rospy.Time()
        self.camera_point.header.stamp = self.world_point.header.stamp
        self.camera_point = self.listener.transformPoint("camera", self.world_point)
        #print(self.camera_point.point.x)
        #print(self.camera_point.point.y)
        #print(self.camera_point.point.z)
        self.cam_model.fromCameraInfo(self.camera_param)
        pixel = [0, 0]
        pixel = list(self.cam_model.project3dToPixel((self.camera_point.point.x, self.camera_point.point.y, self.camera_point.point.z)))
        #print("before " + str(pixel))
        img_height = 234
        if pixel[0] > img_height:
            pixel[0] = img_height
        if pixel[1] > img_height:
            pixel[1] = img_height
        #print("after " + str(pixel))
        pixel = np.around(pixel, decimals=0)
        return list(np.array(pixel).astype(float)/img_height)

    def _env_setup(self, initial_qpos):

        # Called by intializing of task env in order to
        # 1) Unpause sim
        # 2)go to initial position

        #print("Init End Effector Position:" + str(initial_qpos))

        self.gazebo.unpauseSim()

        # Move end effector into position.
        gripper_target = np.array([0.498, 0.005, 0.431])
        gripper_rotation = np.array([0])
        action = np.concatenate([gripper_target, gripper_rotation, 1], None)
        self.set_trajectory_ee(action)
        self.randomize_env.publish()
        self.spawn_objects.publish()
        # init_model_states_all = self.get_model_states()
        # while(len(init_model_states_all) != 5):
        #     init_model_states_all = self.get_model_states()
        #     pass
        # self.init_model_states = {model: position[:] for model, position in init_model_states_all.items() } #mafrod naraga3 x,y,z then passed to 3d to pixel to get u,v
        # self.goal = self._sample_goal()
        #self._get_obs()


    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        #This should include intilization of different objects in the env getting their poses using
        self.delete_objects.publish()
        self.randomize_env.publish()
        init_model_states_all = self.get_model_states()
        while(len(init_model_states_all) != 5):
            init_model_states_all = self.get_model_states()
            pass
        self.init_model_states = {model: position[:] for model, position in init_model_states_all.items()}#same solution as above x,y,z
        self.goal = self._sample_goal()  #3mora l3b hna
        self._get_obs()

