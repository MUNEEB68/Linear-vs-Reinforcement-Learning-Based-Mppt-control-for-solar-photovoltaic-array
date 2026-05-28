import matlab.engine

print("1. Connecting to MATLAB...")
eng = matlab.engine.start_matlab()
model = 'taha_project_trail1'

print("2. Loading model...")
eng.eval(f"load_system('{model}')", nargout=0)
eng.set_param(model, 'SimulationMode', 'normal', nargout=0)

print("3. Starting Simulation...")
eng.set_param(model, 'SimulationCommand', 'start', nargout=0)
eng.set_param(model, 'SimulationCommand', 'pause', nargout=0)

print("4. Attempting a Step...")
eng.set_param(model, 'SimulationCommand', 'step', nargout=0)

print("5. Checking Simulation Status...")
status = eng.get_param(model, 'SimulationStatus')
print(f"Status is: {status}")

print("6. Reading Observation Port (Intentionally Unsafe)...")
# If Simulink has crashed or the port is unreadable, this line will explode and tell us why
obs = eng.eval(f"get_param('{model}/Observation_Out', 'RuntimeObject').InputPort(1).Data")

print(f"SUCCESS! Raw Observation Data: {obs}")
eng.quit()