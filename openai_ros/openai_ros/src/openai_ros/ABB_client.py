#! /usr/bin/env python


import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection
# https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from theconstruct_msgs.msg import RLExperimentInfo
import numpy
import rospy
from std_msgs.msg import Float64
from abb_catkin.srv import JointTraj, JointTrajRequest, EeRpy, EeRpyRequest
import rospy, sys
from controller_manager import controller_manager_interface
from controller_manager_msgs.srv import *


class RobotGazeboEnv(gym.GoalEnv):
    def __init__(self):
    	self.gazebo = GazeboConnection(start_init_physics_parameters=False, reset_world_or_sim="SIMULATION")
        #self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)
    	self.MyControllers = [ 'arm_controller', 'joint_state_controller' ]
    	self.switch_controller = rospy.ServiceProxy('controller_manager/switch_controller', SwitchController)
    	self.list_controllers = rospy.ServiceProxy('controller_manager/list_controllers', ListControllers)
    	self.switch_controller.wait_for_service()
    	self.list_controllers.wait_for_service()
    	self.reset_controls=True

    def sp():

        controller_manager_interface.load_controller(self.MyControllers)
        controller_manager_interface.start_controller(self.MyControllers)

    def unspload(self):

        inhibited = set()

        # to_inhibit = set(rospy.myargv())
        to_inhibit = set(self.MyControllers)

        def stop_controllers(*argc):
            to_stop = to_inhibit & set(cs.name for cs in self.list_controllers().controller if cs.state == 'running')

            if to_stop:
                rospy.logout("Inhibiting controllers: %s" % ', '.join(to_stop))
                self.switch_controller(strictness=SwitchControllerRequest.BEST_EFFORT,
                                  stop_controllers=list(to_stop))
                inhibited.update(to_stop)

        stop_controllers()
        timer = rospy.Timer(rospy.Duration(3), stop_controllers)

        ##hna unload
        controller_manager_interface.unload_controller(self.MyControllers)

        def restart_controllers():
            timer.shutdown()
            # Re-starts inhibited controllers
            self.switch_controller(strictness=SwitchControllerRequest.BEST_EFFORT,
                              start_controllers=list(inhibited))

        restart_controllers()

    # rospy.on_shutdown(restart_controllers)
    # rospy.spin()
    # sys.exit(1)

    def _reset_sim(self):
        """Resets a simulation
        """
        self.gazebo.unpauseSim()
            # self.controllers_object.reset_controllers()
            # self._check_all_systems_ready()
            # self._set_init_pose()
        self.gazebo.pauseSim()
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()
        self.unspload()
        self.spload()
        self.controllers_object.reset_controllers()
        self._check_all_systems_ready()
        self.gazebo.pauseSim()

        return True

    def reset(self):
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        print("Entered reset")
        self._reset_sim()

    def set_trajectory_joints(self, initial_qpos):
        # Set up a trajectory message to publish.

        print("Entered Gazebo Env")
        self.gazebo = GazeboConnection(start_init_physics_parameters=True, reset_world_or_sim="SIMULATION")
        # self.controllers_object = ControllersConnection(namespace="", controllers_list=["joint_state_controller", "arm_controller"])
        # self.reset_controls = True
        # print (self.reset_controls)
        self.seed()

        # Set up ROS related variables
        self.episode_num = 0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)

        print("Exit Gazebo Env")

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

    def get_ee_rpy():

        gripper_rpy_req = EeRpyRequest()
        ee_rpy_client = rospy.ServiceProxy('/ee_rpy_srv', EeRpy)
        gripper_rpy = ee_rpy_client(gripper_rpy_req)

        return gripper_rpy


if __name__ == "__main__":
    print(set_trajectory_joints([ 0, 0, 0, 0, 0, 0 ]))
    print(get_ee_rpy())
