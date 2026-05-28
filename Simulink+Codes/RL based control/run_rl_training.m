%% Differential Reward Training - 300 Episodes
% Goal: 100kW Peak Tracking via Impedance Matching
clear all; clc;

% 1. Model Setup
mdl = 'taha_project_trail1';
if ~bdIsLoaded(mdl); load_system(mdl); end
set_param(mdl, 'SimulationMode', 'accelerator');

% 2. Observations (V, I, P) and Actions (Duty Cycle)
obsInfo = rlNumericSpec([3 1], 'Name', 'observations');
actInfo = rlNumericSpec([1 1], 'LowerLimit', 0, 'UpperLimit', 1, 'Name', 'duty_cycle');

% 3. DDPG Actor/Critic Networks
% Critic
statePath = [featureInputLayer(3,'Name','obs'), fullyConnectedLayer(128), reluLayer, fullyConnectedLayer(64,'Name','sp')];
actionPath = [featureInputLayer(1,'Name','act'), fullyConnectedLayer(64,'Name','ap')];
commonPath = [additionLayer(2,'Name','add'), reluLayer, fullyConnectedLayer(1,'Name','out')];
criticNet = dlnetwork();
criticNet = addLayers(criticNet,statePath); criticNet = addLayers(criticNet,actionPath); criticNet = addLayers(criticNet,commonPath);
criticNet = connectLayers(criticNet,'sp','add/in1'); criticNet = connectLayers(criticNet,'ap','add/in2');
critic = rlQValueFunction(criticNet,obsInfo,actInfo,'ObservationInputNames','obs','ActionInputNames','act');

% Actor (Sigmoid limits D between 0 and 1)
actorNet = [featureInputLayer(3,'Name','obs'), fullyConnectedLayer(128), reluLayer, ...
            fullyConnectedLayer(64), reluLayer, fullyConnectedLayer(1), sigmoidLayer];
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

% 4. Agent Configuration
agentOpts = rlDDPGAgentOptions('SampleTime',1e-3, 'MiniBatchSize', 256, 'ExperienceBufferLength', 1e6);
agentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
agent = rlDDPGAgent(actor, critic, agentOpts);

% 5. Environment & Reset Function (Randomized for 15C/1000W scenarios)
env = rlSimulinkEnv(mdl, [mdl '/RL Agent'], obsInfo, actInfo);
env.ResetFcn = @(in) in.setVariable('Irradiance_init', 400 + rand*600) ...
                       .setVariable('T_init', 10 + rand*20);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 300, ...
    'MaxStepsPerEpisode', 500, ...
    'Plots', 'training-progress', ...
    'StopTrainingValue', 1e12, ...    % Set this very high!
    'ScoreAveragingWindowLength', 10, ...
    'UseParallel', true);

% Enable GPU for matrix math
try trainOpts.Accelerator = 'gpu'; catch; end

% 7. Execution
disp('Starting High-Power MPPT Training...');
trainingStats = train(agent, env, trainOpts);
save('HighPower_MPPT_Agent.mat', 'agent');