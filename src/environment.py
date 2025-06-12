"""
Robotic Arm Environment for Reinforcement Learning

This module creates a simulated robotic arm environment using PyBullet
for reinforcement learning experiments.
"""

import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import time
import math

class RobotArmEnv(gym.Env):
    """
    A robotic arm environment for reinforcement learning.
    """
    
    def __init__(self, task='reaching', render_mode='human', max_steps=200):
        """
        Initialize the robotic arm environment.
        
        Args:
            task (str): Task type ('reaching', 'grasping', 'stacking')
            render_mode (str): Rendering mode ('human', 'rgb_array', None)
            max_steps (int): Maximum steps per episode
        """
        super(RobotArmEnv, self).__init__()
        
        self.task = task
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Connect to PyBullet
        if render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Robot arm parameters
        self.num_joints = 6
        self.joint_limits = [(-np.pi, np.pi)] * self.num_joints
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(self.num_joints * 2)  # +/- for each joint
        
        # Observation: joint positions + end effector position + target position
        obs_dim = self.num_joints + 3 + 3  # joints + ee_pos + target_pos
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            np.array: Initial observation
        """
        super().reset(seed=seed)
        
        # Clear the simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot arm (using a simple URDF or create programmatically)
        self.robot_id = self._create_robot_arm()
        
        # Set initial joint positions
        self.initial_joint_positions = [0, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0]
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, self.initial_joint_positions[i])
        
        # Create target
        self.target_position = self._generate_target_position()
        self.target_id = p.loadURDF("sphere_small.urdf", self.target_position)
        p.changeVisualShape(self.target_id, -1, rgbaColor=[1, 0, 0, 0.8])
        
        # Create object for grasping tasks
        if self.task in ['grasping', 'stacking']:
            self.object_position = self._generate_object_position()
            self.object_id = p.loadURDF("cube_small.urdf", self.object_position)
            p.changeVisualShape(self.object_id, -1, rgbaColor=[0, 1, 0, 0.8])
        
        self.current_step = 0
        
        # Step simulation to stabilize
        for _ in range(100):
            p.stepSimulation()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): Action to take
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        # Convert discrete action to joint movement
        joint_idx = action // 2
        direction = 1 if action % 2 == 0 else -1
        
        # Get current joint position
        joint_state = p.getJointState(self.robot_id, joint_idx)
        current_position = joint_state[0]
        
        # Calculate new position
        step_size = 0.1  # radians
        new_position = current_position + direction * step_size
        
        # Apply joint limits
        min_limit, max_limit = self.joint_limits[joint_idx]
        new_position = np.clip(new_position, min_limit, max_limit)
        
        # Set joint position
        p.setJointMotorControl2(
            self.robot_id,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=new_position,
            force=500
        )
        
        # Step simulation
        p.stepSimulation()
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        info = {
            'success': self._is_success(),
            'distance_to_target': self._get_distance_to_target()
        }
        
        return observation, reward, done, truncated, info
    
    def _create_robot_arm(self):
        """
        Create a simple robotic arm programmatically.
        
        Returns:
            int: Robot body ID
        """
        # Base
        base_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.1)
        base_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.1, height=0.1, rgbaColor=[0.5, 0.5, 0.5, 1])
        
        # Links
        link_masses = [1.0] * self.num_joints
        link_collision_shapes = []
        link_visual_shapes = []
        link_positions = []
        link_orientations = []
        link_inertial_frame_positions = []
        link_inertial_frame_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []
        
        link_lengths = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        
        for i in range(self.num_joints):
            # Collision shape
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, 
                radius=0.02, 
                height=link_lengths[i]
            )
            link_collision_shapes.append(collision_shape)
            
            # Visual shape
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=0.02, 
                length=link_lengths[i],
                rgbaColor=[0.8, 0.4, 0.2, 1]
            )
            link_visual_shapes.append(visual_shape)
            
            # Link position (relative to parent)
            if i == 0:
                link_positions.append([0, 0, 0.1])  # First link from base
            else:
                link_positions.append([0, 0, link_lengths[i-1]])
            
            link_orientations.append([0, 0, 0, 1])
            link_inertial_frame_positions.append([0, 0, link_lengths[i]/2])
            link_inertial_frame_orientations.append([0, 0, 0, 1])
            link_parent_indices.append(i)
            link_joint_types.append(p.JOINT_REVOLUTE)
            
            # Alternate joint axes for more realistic arm
            if i % 2 == 0:
                link_joint_axes.append([0, 0, 1])  # Z-axis rotation
            else:
                link_joint_axes.append([1, 0, 0])  # X-axis rotation
        
        # Create multi-body
        robot_id = p.createMultiBody(
            baseMass=2.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[0, 0, 0],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_frame_positions,
            linkInertialFrameOrientations=link_inertial_frame_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes
        )
        
        return robot_id
    
    def _get_observation(self):
        """
        Get current observation.
        
        Returns:
            np.array: Current state observation
        """
        # Get joint positions
        joint_positions = []
        for i in range(self.num_joints):
            joint_state = p.getJointState(self.robot_id, i)
            joint_positions.append(joint_state[0])
        
        # Get end effector position
        ee_state = p.getLinkState(self.robot_id, self.num_joints - 1)
        ee_position = ee_state[0]
        
        # Combine observation
        observation = np.concatenate([
            joint_positions,
            ee_position,
            self.target_position
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward(self):
        """
        Calculate reward for current state.
        
        Returns:
            float: Reward value
        """
        # Get end effector position
        ee_state = p.getLinkState(self.robot_id, self.num_joints - 1)
        ee_position = np.array(ee_state[0])
        target_position = np.array(self.target_position)
        
        # Distance-based reward
        distance = np.linalg.norm(ee_position - target_position)
        
        if self.task == 'reaching':
            # Reward for getting close to target
            reward = -distance
            
            # Bonus for reaching target
            if distance < 0.05:
                reward += 10.0
                
        elif self.task == 'grasping':
            # Additional logic for grasping
            reward = -distance
            
            # Check if object is grasped (simplified)
            if hasattr(self, 'object_id'):
                object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
                object_distance = np.linalg.norm(ee_position - np.array(object_pos))
                if object_distance < 0.1:
                    reward += 5.0
        
        else:  # stacking
            reward = -distance
        
        # Penalty for large joint movements (smoothness)
        joint_velocities = []
        for i in range(self.num_joints):
            joint_state = p.getJointState(self.robot_id, i)
            joint_velocities.append(abs(joint_state[1]))
        
        velocity_penalty = -0.01 * sum(joint_velocities)
        reward += velocity_penalty
        
        return reward
    
    def _is_done(self):
        """
        Check if episode is done.
        
        Returns:
            bool: True if episode should end
        """
        return self._is_success()
    
    def _is_success(self):
        """
        Check if task is successfully completed.
        
        Returns:
            bool: True if task is successful
        """
        distance = self._get_distance_to_target()
        return distance < 0.05
    
    def _get_distance_to_target(self):
        """
        Get distance from end effector to target.
        
        Returns:
            float: Distance to target
        """
        ee_state = p.getLinkState(self.robot_id, self.num_joints - 1)
        ee_position = np.array(ee_state[0])
        target_position = np.array(self.target_position)
        return np.linalg.norm(ee_position - target_position)
    
    def _generate_target_position(self):
        """
        Generate a random target position within reach.
        
        Returns:
            list: Target position [x, y, z]
        """
        # Generate target within arm's reach
        radius = np.random.uniform(0.3, 0.8)
        angle = np.random.uniform(0, 2 * np.pi)
        height = np.random.uniform(0.2, 0.6)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        
        return [x, y, z]
    
    def _generate_object_position(self):
        """
        Generate object position for grasping tasks.
        
        Returns:
            list: Object position [x, y, z]
        """
        # Place object near the target
        offset = np.random.uniform(-0.1, 0.1, 3)
        object_pos = np.array(self.target_position) + offset
        object_pos[2] = max(object_pos[2], 0.1)  # Ensure object is above ground
        
        return object_pos.tolist()
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            # PyBullet GUI is already rendering
            time.sleep(1./240.)  # Control rendering speed
        elif mode == 'rgb_array':
            # Get camera image
            view_matrix = p.computeViewMatrix([1, 1, 1], [0, 0, 0], [0, 0, 1])
            proj_matrix = p.computeProjectionMatrixFOV(60, 1, 0.1, 100)
            
            width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                width=640, height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )
            
            rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
            return rgb_array
    
    def close(self):
        """
        Close the environment.
        """
        p.disconnect(self.physics_client)

# Simple environment for testing without PyBullet
class SimpleRobotArmEnv(gym.Env):
    """
    A simplified robotic arm environment for testing without PyBullet.
    """
    
    def __init__(self, num_joints=3):
        super(SimpleRobotArmEnv, self).__init__()
        
        self.num_joints = num_joints
        self.joint_positions = np.zeros(num_joints)
        self.target_position = np.array([1.0, 0.0])
        
        # Action space: discrete actions for each joint
        self.action_space = spaces.Discrete(num_joints * 2)
        
        # Observation space: joint positions + end effector position + target position
        obs_dim = num_joints + 2 + 2  # joints + ee_pos + target_pos
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.max_steps = 200
        self.current_step = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.joint_positions = np.random.uniform(-np.pi/4, np.pi/4, self.num_joints)
        self.target_position = np.random.uniform(-1.5, 1.5, 2)
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        joint_idx = action // 2
        direction = 1 if action % 2 == 0 else -1
        
        # Update joint position
        step_size = 0.1
        self.joint_positions[joint_idx] += direction * step_size
        self.joint_positions[joint_idx] = np.clip(self.joint_positions[joint_idx], -np.pi, np.pi)
        
        # Calculate end effector position (simplified 2D kinematics)
        ee_pos = self._forward_kinematics()
        
        # Calculate reward
        distance = np.linalg.norm(ee_pos - self.target_position)
        reward = -distance
        
        if distance < 0.1:
            reward += 10.0
            done = True
        else:
            done = False
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        info = {'success': distance < 0.1, 'distance': distance}
        
        return self._get_observation(), reward, done, truncated, info
    
    def _forward_kinematics(self):
        """Calculate end effector position using forward kinematics."""
        x, y = 0, 0
        angle = 0
        link_length = 0.5
        
        for joint_angle in self.joint_positions:
            angle += joint_angle
            x += link_length * np.cos(angle)
            y += link_length * np.sin(angle)
        
        return np.array([x, y])
    
    def _get_observation(self):
        ee_pos = self._forward_kinematics()
        return np.concatenate([
            self.joint_positions,
            ee_pos,
            self.target_position
        ]).astype(np.float32)
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass

