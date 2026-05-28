import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matlab.engine
import time

class SolarSimulinkEnv(gym.Env):
    def __init__(self):
        super(SolarSimulinkEnv, self).__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        print("Starting MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()
        self.model_name = 'taha_project_trail1'
        self.eng.eval(f"load_system('{self.model_name}')", nargout=0)
        
        # Match your Powergui exactly
        self.eng.set_param(self.model_name, 'SolverType', 'Fixed-step', nargout=0)
        self.eng.set_param(self.model_name, 'FixedStep', '5e-6', nargout=0) 
        self.eng.set_param(self.model_name, 'StopTime', '0.05', nargout=0) 
        self.eng.set_param(self.model_name, 'SimulationMode', 'normal', nargout=0)

    def _get_sensor_data(self):
        """Extracts data safely using MATLAB workspace variables"""
        try:
            # We tell MATLAB to grab the data and put it in a workspace variable 'v' and 'r'
            self.eng.eval(f"try, v = get_param('{self.model_name}/Observation_Out','RuntimeObject').InputPort(1).Data; catch, v = [0,0,0]; end", nargout=0)
            self.eng.eval(f"try, r = get_param('{self.model_name}/Reward_Out','RuntimeObject').InputPort(1).Data; catch, r = 0; end", nargout=0)
            
            obs = np.array(self.eng.workspace['v'], dtype=np.float32).flatten()
            reward = float(np.array(self.eng.workspace['r']).flatten()[0])
            return obs, reward
        except:
            return np.zeros(3, dtype=np.float32), 0.0

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Force a hard stop and restart
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        # Tiny delay to let MATLAB breath
        time.sleep(0.1) 
        
        # Start the model
        self.eng.set_param(self.model_name, 'SimulationCommand', 'start', nargout=0)
        self.eng.set_param(self.model_name, 'SimulationCommand', 'pause', nargout=0)
        
        obs, _ = self._get_sensor_data()
        return obs, {}

    def step(self, action):
        duty_cycle = float(np.clip(action[0], 0.0, 1.0))
        self.eng.set_param(f'{self.model_name}/Duty_Cycle_In', 'Value', str(duty_cycle), nargout=0)
        
        # Take the step
        self.eng.set_param(self.model_name, 'SimulationCommand', 'step', nargout=0)
        
        obs, reward = self._get_sensor_data()
        
        # Check if actually done
        status = self.eng.get_param(self.model_name, 'SimulationStatus')
        done = (status == 'stopped')
        
        return obs, float(reward), done, False, {}