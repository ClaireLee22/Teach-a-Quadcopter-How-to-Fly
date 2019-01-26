from task import Task
import numpy as np


class TakeOff(Task):
    """A task to lift off the ground and reach a target height."""
    def __init__(self, init_pose= np.array([0., 0., 0.1, 0., 0., 0.]), init_velocities=None, 
                 init_angle_velocities=None, runtime=5., target_pos=np.array([0., 0., 10.])):
        """Initialize a TakeOff object."""
        # invoking the __init__ of the parent class  
        Task.__init__(self, init_pose, init_velocities, 
                      init_angle_velocities, runtime, target_pos)
        
 
       
    def get_reward(self, rotor_speeds, old_pose, reward_function_choice):
        """Uses current pose of sim to return reward."""
        z_distance_from_target = self.sim.pose[2] - self.target_pos[2]
        reward = 0
        #####################
        # Reward Function 1 #
        #####################
        if reward_function_choice == 1:
            if self.sim.pose[2] > old_pose[2]:
                reward = 1
                if  self.sim.pose[2] > self.target_pos[2]:
                    reward = -0.05
            else:
                reward = -0.5
        
        #####################
        # Reward Function 2 #
        #####################
        if reward_function_choice == 2:
            if self.sim.pose[2] > old_pose[2]:
                reward = 0.02-.001*abs(z_distance_from_target) + .001*self.sim.pose[2]
            else:
                reward = 0.02-.001*abs(z_distance_from_target) - .0001

            
        ###########################
        # For track train process #
        ###########################
        #num_timesteps = self.sim.time/self.sim.dt
        #print('num_timesteps', num_timesteps)
        #print('self.sim.pose[2]', self.sim.pose[2])
        #print('old pose[2]', old_pose[2])
        #print('roter speed ',rotor_speeds)
        #print('reward ', reward)
        #print("np.tanh(reward) ", np.tanh(reward))
        #print()
        
        return reward
       
    
    def step(self, rotor_speeds, reward_function_choice):
        """Uses action to obtain next state, reward, done."""
        old_pose = self.sim.pose
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds, old_pose, reward_function_choice)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
