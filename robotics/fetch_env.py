#import cv2
import numpy as np
from gym.envs.robotics import rotations, utils, robot_env
import random

def goal_distance(goal_a, goal_b):  # 3mora l3b hna

    if len(goal_a.shape) > len(goal_b.shape):

        i = np.where(goal_a[ :, 3: ] == goal_b[ 3: ])
        return np.around(np.linalg.norm(goal_a[ i[ 0 ][ 0 ] ] - goal_b), decimals=3)
    else:

        i = np.where(goal_b[ :, 3: ] == goal_a[ 3: ])
        return np.around(np.linalg.norm(goal_a - goal_b[ i[ 0 ][ 0 ] ]), decimals=3)

# def goal_distance(goal_a, goal_b):
#     assert goal_a.shape == goal_b.shape
#     return np.linalg.norm(goal_a - goal_b, axis=-1)

def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis = axis/ np.linalg.norm(axis)
    #print(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat = quat/ np.linalg.norm(quat)
    return quat

class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.init_objects_states = dict()
        #self.i = 0 #moemen added
        self.objects = [ "object0", "object1", "object2", "object3", "object4" ]

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=5,
            initial_qpos=initial_qpos) #

    # GoalEnv methods
    # ----------------------------

    # added our reward
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        #print(self.reward_type)
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            c = np.array(d < self.distance_threshold) #can be squeezed if needed check
            c = float(np.where(c == False, -0.05, 1))
            return c
        else:
            return -5
    # def compute_reward(self, achieved_goal, goal, info):
    #     # Compute distance between goal and the achieved goal.
    #     d = goal_distance(achieved_goal, goal)
    #     if self.reward_type == 'sparse':
    #         return -(d > self.distance_threshold).astype(np.float32)
    #     else:
    #         return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()
    
    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl ,gripper_ctrl = action[:3], action[3], action[4]
        #pos_ctrl[1:]=np.array([0, 0])
        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl *= 90.
        #rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        axis = np.array([ 1, 0, 0 ])
               
        rot_ctrl = quat_from_angle_and_axis(np.pi * rot_ctrl / 180, axis)
        rot_ctrl = rotations.quat_mul(np.array([1., 0., 1., 0.]), rot_ctrl)
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        
        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)


    def _get_obs(self):
        #get image
        rgb_image, depth_image = self.render(mode='rgb_array', width=self.width, height=self.height) # added
        #self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = np.asarray(rgb_image, dtype = np.float32)/256 # added, cast image from tuple to np.array
        depth_image = np.asarray(depth_image, dtype = np.float32).reshape(200,200,1) # added cast depth from tuple to np.array, reshape from 200*200 to200*200*1
        stacked_image = np.concatenate([rgb_image, depth_image], axis=-1) # added, create stcked image

        #cv2.imwrite("renders/timestep{0}.png".format(self.i),self.rgb_image) #added #optional, save renders to folder
        #self.i += 1 #added #optional

        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        if self.has_object:
            
            object_pos = []
            for obj in self.objects:
                object_pos.append(np.concatenate((self.sim.data.get_site_xpos(obj),
                                                  self._return_pixel(list(self.init_objects_states[obj])))))
            object_pos = np.array(object_pos)
            assert object_pos.shape == (5, 5)
            # rotations
            # object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            # object_velp = self.sim.data.get_site_xvelp('object0') * dt
            # object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            # object_rel_pos = object_pos - grip_pos
            # object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            #squeezing does nothing here
            achieved_goal = np.squeeze(object_pos.copy())

        # adjusted the state to include the position of the gripper, wether it's closed or open
        # and the velocity of that gripper should i delete the grip velocity ?
        state = np.concatenate([
            grip_pos, gripper_state, grip_velp
        ])
        assert state.shape == (8,)

        return {
            'image': stacked_image.copy(),
            'achieved': achieved_goal.copy(),
            'desired': self.goal.copy(),
            'state': state.copy(),

        }

    def _viewer_setup(self):
        #setup the camera/viewer location
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.2 #2.5
        self.viewer.cam.azimuth = 180 #132.
        self.viewer.cam.elevation = -40 #-14.

        #added transformation matrix calculations
        elevation = np.abs(self.viewer.cam.elevation)
        self.camera_position = lookat + [self.viewer.cam.distance*np.cos(elevation*np.pi/180), 0, self.viewer.cam.distance*np.sin(elevation*np.pi/180)]
        fovy = self.sim.model.cam_fovy[self.camera_id]
        f = 0.5 * self.height / np.tan(fovy * np.pi / 360)
        self.intrinsic_matrix = np.array(((f, 0, self.width / 2), (0, f, self.height/2),(0,0,1)))
        T_cw_inv = np.array(((0,    np.cos(90-elevation),  np.cos(-elevation),   self.camera_position[0]),
                             (1,    0,                     0,                    self.camera_position[1]),
                             (0,    np.cos(-elevation),    np.cos(90+elevation), self.camera_position[2]),
                             (0,    0,                     0,                      1)))
        T_wc = np.linalg.inv(T_cw_inv)
        self.extrinsic_matrix = T_wc[:3,:]

    def _return_pixel(self, object_position):
        object_position.append(1)
        position = np.array(object_position).reshape(1,4)
        result = np.matmul(self.extrinsic_matrix, position.T)
        result /= result[2]
        pixels = np.matmul(self.intrinsic_matrix, result)
        pixels = np.clip(np.around(pixels),0,200).astype(float)/200
        return pixels[:2].reshape(1,2)[0,:]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            #object_xpos = self.initial_gripper_xpos[:2]
            for obj in self.objects:
                object_xpos = self.initial_gripper_xpos[ :2 ]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                object_qpos = self.sim.data.get_joint_qpos(obj+':joint')
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos(obj+':joint', object_qpos)

        self.sim.forward()

        if self.has_object:
            for obj in self.objects:
                self.init_objects_states[obj] = self.sim.data.get_site_xpos(obj).copy()

        return True

    def _sample_goal(self):
        #this function should be modified to be one of the objects pose in addition to any 3d position in the
        # vicinity with placing position 50% of the times in air

        #print("sample goal")
        #get the xyz position of a random object in the world
        if self.has_object:
            _, position = random.choice(list(self.init_objects_states.items()))

            goal_position = position + np.random.uniform(-self.target_range, self.target_range, size=3)
            #goal_position[0] = np.clip(goal_position[0], 0, 1.09)
            #goal_position[1] = np.clip(goal_position[1], -0.49, 0.49)
            #goal_position[0] = np.abs(goal_position[0])
            goal_position += self.target_offset
            goal_position[2] = position[2]

            #get the pixel location of the object in the image
            pixel_position = self._return_pixel(list(position))
            goal = np.concatenate([goal_position, pixel_position])
            if self.target_in_the_air and np.random.uniform() < 0.5:
               goal[2] += np.random.uniform(0, 0.45)
            #goal[:3]=np.around(goal[:3],decimals=3)
            #print('goal = ' + str(goal))
        else:
            goal = self.initial_gripper_xpos[ :3 ] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    # def _sample_goal(self):
    #     if self.has_object:
    #         goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
    #         goal += self.target_offset
    #         goal[2] = self.height_offset
    #         if self.target_in_the_air and self.np_random.uniform() < 0.5:
    #             goal[2] += self.np_random.uniform(0, 0.45)
    #     else:
    #         goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    #     return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            for obj in self.objects:
                self.init_objects_states[obj] = self.sim.data.get_site_xpos(obj).copy()
                
            #self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
