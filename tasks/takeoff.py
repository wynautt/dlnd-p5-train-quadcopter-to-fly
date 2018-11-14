import numpy as np
import random
from physics_sim import PhysicsSim


class Takeoff:
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, target_z=10., runtime=5.):
        """Initialize a Task object.
        Params
        ======
            runtime: time limit for each episode
            target_z: target/goal z position for the agent
        """

        self.runtime = runtime
        self.init_sim()
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_z = target_z if target_z is not None else 10.

    def init_sim(self):
        env_bounds = 150.0
        x = random.uniform(-env_bounds, env_bounds)
        y = random.uniform(-env_bounds, env_bounds)

        init_pose = np.array([x, y, 0.0, 0.0, 0.0, 0.0])
        init_velocities = np.array([0.0, 0.0, 0.0])
        init_angle_velocities = np.array([0.0, 0.0, 0.0])

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, self.runtime)

    def get_reward(self, done):
        if done and self.sim.time < self.sim.runtime:
            reward = -1  # penalize crash
        else:
            z = self.sim.pose[2]
            v_z = self.sim.v[2]
            steps = self.sim.time / self.sim.dt
            # penalize distance to target_z + reward positive z velocity + reward keep flying
            reward = 1.0 - 0.03 * abs(self.target_z - z) + 0.01 * v_z + 0.001 * steps
            reward = np.tanh(reward)  # reward clipping to avoid exploding gradients
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        done = False
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward(done)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        # self.sim.reset()
        self.init_sim()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
