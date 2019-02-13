#! /usr/bin/env python


import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection
#https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from theconstruct_msgs.msg import RLExperimentInfo
import numpy
import rospy
from std_msgs.msg import Float64
from abb_catkin.srv import JointTraj, JointTrajRequest, EeRpy, EeRpyRequest

class RobotGazeboEnv(gym.GoalEnv):

    def set_trajectory_joints(self, initial_qpos):
        # Set up a trajectory message to publish.

        print ("Entered Gazebo Env")

        #self.gazebo = GazeboConnection(start_init_physics_parameters=False, reset_world_or_sim="SIMULATION")
        #self.controllers_object = ControllersConnection(namespace="", controllers_list=["joint_state_controller", "arm_controller"])
        #self.reset_controls = True
        #print (self.reset_controls)
        #self.seed()

            # Set up ROS related variables
        #self.episode_num = 0
        #self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        #print("Exit Gazebo Env")


        joint_point = JointTrajRequest()

        joint_point.point.positions = [ None ] * 6
        joint_point.point.positions[ 0 ] = initial_qpos[ 0 ]
        joint_point.point.positions[ 1 ] = initial_qpos[ 1 ]
        joint_point.point.positions[ 2 ] = initial_qpos[ 2 ]
        joint_point.point.positions[ 3 ] = initial_qpos[ 3 ]
        joint_point.point.positions[ 4 ] = initial_qpos[ 4 ]
        joint_point.point.positions[ 5 ] = initial_qpos[ 5 ]
        joint_traj_client = rospy.ServiceProxy('/joint_traj_srv', JointTraj)
        result = joint_traj_client(joint_point)
        print(True)

        return True

    def get_ee_rpy(self):

        gripper_rpy_req = EeRpyRequest()
        ee_rpy_client = rospy.ServiceProxy('/ee_rpy_srv', EeRpy)
        gripper_rpy = ee_rpy_client(gripper_rpy_req)

        return gripper_rpy
    

if __name__ == "__main__":
    print(set_trajectory_joints([ 0, 0, 0, 0, 0, 0 ]))
    print(get_ee_rpy())
