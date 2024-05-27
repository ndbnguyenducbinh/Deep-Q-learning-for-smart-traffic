# ---- Import Section ----

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), 'tools'))
import traci
import joblib
from traci._trafficlight import Logic, Phase # Class Logic, Phase for changing TFL phase
from itertools import repeat, chain # For repeat time_steps

import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



# Load model
current_weekday = 1
model_car = joblib.load('/media/binhnguyenduc/ubuntu_data/vscode/Simulate_sumo/models/model_car.pkl')
model_bike = joblib.load('/media/binhnguyenduc/ubuntu_data/vscode/Simulate_sumo/models/model_bike.pkl')
model_bus = joblib.load('/media/binhnguyenduc/ubuntu_data/vscode/Simulate_sumo/models/model_bus.pkl')
model_truck = joblib.load('/media/binhnguyenduc/ubuntu_data/vscode/Simulate_sumo/models/model_truck.pkl')
model_classification = joblib.load('/media/binhnguyenduc/ubuntu_data/vscode/Simulate_sumo/models/stacking_classifier.joblib')



# Function to convert time to seconds
def time_to_seconds(time_obj):
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds

# Function to predict traffic in real-time
def predict_realtime(current_time, current_weekday, car_count, bike_count, bus_count, truck_count):
    input_data = pd.DataFrame({
        'Time': [current_time],
        'sin_day': [np.sin(2 * np.pi * current_weekday / 7)],
        'cos_day': [np.cos(2 * np.pi * current_weekday / 7)],
        'CarCount': [car_count],
        'BikeCount': [bike_count],
        'BusCount': [bus_count],
        'TruckCount': [truck_count]
    })

    car_pred = model_car.predict(input_data[['Time', 'sin_day', 'cos_day']])
    bike_pred = model_bike.predict(input_data[['Time', 'sin_day', 'cos_day']])
    bus_pred = model_bus.predict(input_data[['Time', 'sin_day', 'cos_day']])
    truck_pred = model_truck.predict(input_data[['Time', 'sin_day', 'cos_day']])

    traffic_pred = model_classification.predict(input_data)
    traffic_pred = traffic_pred.ravel()

    return car_pred, bike_pred, bus_pred, truck_pred, traffic_pred


'''tl_logic_ids = {
    0: "tlLogic0",
    1: "tlLogic1",
    2: "tlLogic2",
    3: "tlLogic3",
    4: "tlLogic4"
}
'''
'''
def set_initial_phase_duration(tls_id):
    """Sets the initial phase durations for a traffic light."""
    # Get all program logics for the traffic light
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    # Assuming the first logic is the one you want to modify
    logic = logics[0] 
    # Iterate through phases and set durations
    for phase in logic.phases:
        if 'G' in phase.state:
            phase.duration = 42  # Green phase duration
        elif 'y' in phase.state:
            phase.duration = 3   # Yellow phase duration
        else:
            phase.duration = 45  # Red phase duration
    # Update the traffic light program
    traci.trafficlight.setProgramLogic(tls_id, logic)
'''
def adjust_phases(tls_id, increase_green_edges, increase_red_edges, duration_change):
    """Adjusts the green and red phase durations for specific edges."""
    # Get all program logics
    #tls_id = tl_logic_ids[program_id]
    
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    # Assuming you want to modify the first logic
    logic = logics[0]
    # Iterate through phases
    for phase_index, phase in enumerate(logic.phases):
        state = phase.state
        # Check if any edge in the phase needs green time increase
        if any(edge in state for edge in increase_green_edges):
            if 'G' in state:
                phase.duration += duration_change 
        # Check if any edge in the phase needs red time increase
        if any(edge in state for edge in increase_red_edges):
            if 'r' in state:
                phase.duration += duration_change
    # Update the traffic light program
    traci.trafficlight.setProgramLogic(tls_id, logic)

#2 State representation
class TrafficState:
    def __init__(self, edges, history_length=120):
        self.edges = edges
        self.history_length = history_length
        # Initialize data structures for WE and SN  edge
        
        self.we_queue_length_history = deque(maxlen=history_length)
        self.we_vehicle_speed_history = deque(maxlen=history_length) 
        self.we_waiting_time_history = deque(maxlen=history_length) 

        self.sn_queue_length_history = deque(maxlen=history_length)
        self.sn_vehicle_speed_history = deque(maxlen=history_length)
        self.sn_waiting_time_history = deque(maxlen=history_length)

        self.counted_vehicles = {   # For observation
                'counted_vehicles': [set(), set()],
                'car': [set(), set()],
                'motorbike': [set(), set()],
                'bus': [set(), set()],
                'truck': [set(), set()],
                'total': [0.0, 0.0], # For vertical and horizontal lane

                'avg_speed': [0.0, 0.0],
                'avg_waiting_time': [0.0, 0.0],
                'queue_length': [0.0, 0.0], # For vertical and horizontal lane
                }
        self.we_predicted_traffic = 0
        self.sn_predicted_traffic = 0
        
    def update(self):
        current_time = traci.simulation.getTime()
        current_weekday = 1
        for edge in self.edges:
            i = self.edges.index(edge)
            vehicle_ids = traci.edge.getLastStepVehicleIDs(edge)
            queue_length = traci.edge.getLastStepHaltingNumber(edge)
            avg_speed = traci.edge.getLastStepMeanSpeed(edge)
            avg_waiting_time = traci.edge.getWaitingTime(edge)
            self.counted_vehicles['queue_length'][i%2] += queue_length
            self.counted_vehicles['avg_speed'][i%2] += avg_speed
            self.counted_vehicles['avg_waiting_time'][i%2] += avg_waiting_time
            for vehicle_id in vehicle_ids:
                if vehicle_id not in self.counted_vehicles['counted_vehicles'][i%2]:
                    self.counted_vehicles['counted_vehicles'][i%2].add(vehicle_id)
                    vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                    if vehicle_type == 'car':
                        self.counted_vehicles['car'][i%2].add(vehicle_id)
                    elif vehicle_type == 'motorbike':
                        self.counted_vehicles['motorbike'][i%2].add(vehicle_id)
                    elif vehicle_type == 'bus':
                        self.counted_vehicles['bus'][i%2].add(vehicle_id)
                    elif vehicle_type == 'truck':
                        self.counted_vehicles['truck'][i%2].add(vehicle_id)
                    self.counted_vehicles['total'][i%2] += 1
                   
        self.sn_queue_length_history.append(self.counted_vehicles['queue_length'][0])
        self.sn_vehicle_speed_history.append(self.counted_vehicles['avg_speed'][0])
        self.sn_waiting_time_history.append(self.counted_vehicles['avg_waiting_time'][0])

        self.we_queue_length_history.append(self.counted_vehicles['queue_length'][1])
        self.we_vehicle_speed_history.append(self.counted_vehicles['avg_speed'][1])
        self.we_waiting_time_history.append(self.counted_vehicles['avg_waiting_time'][1])

        car_pred_ver, bike_pred_ver, bus_pred_ver, truck_pred_ver, traffic_pred_ver = predict_realtime(
            current_time,
            current_weekday,
            len(self.counted_vehicles['car'][0]),
            len(self.counted_vehicles['motorbike'][0]),
            len(self.counted_vehicles['bus'][0]),
            len(self.counted_vehicles['truck'][0])
        )
        self.sn_predicted_traffic = traffic_pred_ver[0]

        car_pred_hor, bike_pred_hor, bus_pred_hor, truck_pred_hor, traffic_pred_hor = predict_realtime(
            current_time,
            current_weekday,
            len(self.counted_vehicles['car'][1]),
            len(self.counted_vehicles['motorbike'][1]),
            len(self.counted_vehicles['bus'][1]),
            len(self.counted_vehicles['truck'][1])
        )
        self.we_predicted_traffic = traffic_pred_hor[0] #0:low 1 normal 2:high 3:heavy

    def get_state(self):
        #doc la 0,2 ngang la 1,3
        we_total_queue_length = sum(self.we_queue_length_history)
        we_total_vehicle_speed = sum(self.we_vehicle_speed_history)
        we_total_waiting_time = sum(self.we_waiting_time_history)

        sn_total_queue_length = sum(self.sn_queue_length_history)
        sn_total_vehicle_speed = sum(self.sn_vehicle_speed_history)
        sn_total_waiting_time = sum(self.sn_waiting_time_history)

        Predict_traffic_status_we = self.we_predicted_traffic
        Predict_traffic_status_sn = self.sn_predicted_traffic

        print("WE",we_total_queue_length, we_total_vehicle_speed, we_total_waiting_time)
        print("SN",sn_total_queue_length, sn_total_vehicle_speed, sn_total_waiting_time)

        state = np.array([sn_total_queue_length, sn_total_vehicle_speed, sn_total_waiting_time, 
                          we_total_queue_length, we_total_vehicle_speed, we_total_waiting_time, 
                          Predict_traffic_status_sn, Predict_traffic_status_we]) 
        return state


    def reset(self):
        self.counted_vehicles.clear()
        self.we_queue_length_history.clear()
        self.we_vehicle_speed_history.clear()
        self.we_waiting_time_history.clear()
        self.sn_queue_length_history.clear()
        self.sn_vehicle_speed_history.clear()
        self.sn_waiting_time_history.clear()
        self.we_predicted_traffic = 0
        self.sn_predicted_traffic = 0




#3 Deep Q-Network
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        #self.memory = deque(maxlen=memory_size)
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.model = self.NN_build_model()
        
    def NN_build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(self.state_size,), activation='relu'))  # Add input_shape
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))   
    
    def act(self, state):
        # Implement epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        print("state shape: ", state.shape)
        print("input model shape", self.model.input_shape)
        state = np.expand_dims(state, axis=0)
        print("reshape state shape: ", state.shape)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # Sample experiences from memory and train the neural network
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        # Load a trained model
        self.model.load_weights(name)
    
    def save(self, name):
        # Save the trained model
        self.model.save_weights(name)

def calculate_reward(prev_state, action, current_state):
    reward = 0
    ratio_speed = 0.6
    ratio_waiting_time = 0.3
    ratio_queue_length = 0.3
    ratio_traffic_status = 0.1
    
    sn_speed_change = current_state[1] - prev_state[1]
    we_speed_change = current_state[4] - prev_state[4]
    speed_change = we_speed_change + sn_speed_change
    reward += speed_change * ratio_speed

    sn_waiting_time_change = current_state[2] - prev_state[2]
    we_waiting_time_change = current_state[5] - prev_state[5]
    waiting_time_change = we_waiting_time_change + sn_waiting_time_change
    reward -= waiting_time_change * ratio_waiting_time

    sn_queue_length_change = current_state[0] - prev_state[0]
    we_queue_length_change = current_state[3] - prev_state[3]
    queue_length_change = we_queue_length_change + sn_queue_length_change
    reward -= queue_length_change * ratio_queue_length

    sn_traffic_status_change = current_state[6] - prev_state[6]
    we_traffic_status_change = current_state[7] - prev_state[7]
    traffic_status_change = we_traffic_status_change + sn_traffic_status_change
    reward -= traffic_status_change * ratio_traffic_status
    
    return reward




# Define edges of the intersection to observe
edges = ["e0", "e2", "e4", "e6"]

# Time to stop simulate (s)
stop_time = 1000

# Time to start simulate (s)
start_time = 0

# Initialize variables to store total vehicle count each type
count = 0

# DQN parameters
state_size = 8  # Size of your state representation 
action_size = 5  # As defined (5 possible actions)
#memory_size = 2000
batch_size = 32
gamma = 0.95  # Discount factor
learning_rate = 0.01
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# Training settings
num_episodes = 500


# Start SUMO simulation
sumoCmd = ["sumo-gui", "-c", '/media/binhnguyenduc/ubuntu_data/vscode/Simulate_sumo/Env/map_tl_2.sumocfg']
traci.start(sumoCmd, port=8813) 

# ---- Main ----

# Initialize agent
agent = DQNAgent(state_size, action_size)
traffic_state = TrafficState(edges)

train_counter = 0

for episode in range(num_episodes):
    traci.simulationStep()
    #set base time of green phase 42s ...
    #set_initial_phase_duration(tls_id=0)
    traffic_state = TrafficState(edges)

    current_time = 0
    state = traffic_state.get_state()
    
    #we_tls_id = [] # ID of traffic light in WE direction
    #sn_tls_id = [] # ID of traffic light in SN direction

    for time_step in range(stop_time):
        action = agent.act(state)

        if action == 0:
            pass
        elif action == 1: 
            # Increase green phase duration for WE (e0, e4) by 5s
            adjust_phases(tls_id='0', increase_green_edges=["e0", "e4"], 
                          increase_red_edges=["e2", "e6"], duration_change=5)
        elif action == 2: 
            # Increase green phase duration for WE e0 e4 10s
            adjust_phases(tls_id='0', increase_green_edges=["e0", "e4"],
                          increase_red_edges=["e2", "e6"], duration_change=10)
        elif action == 3: 
            # Decrease green phase duration for WE e0 e4 5s
            adjust_phases(tls_id='0', increase_green_edges=["e0", "e4"],
                          increase_red_edges=["e2", "e6"], duration_change=-5)
        elif action == 4: 
            # Decrease green phase duration for WE e0 e4 10s
            adjust_phases(tls_id='0', increase_green_edges=["e0", "e4"],
                          increase_red_edges=["e2", "e6"], duration_change=-10)
        
        next_state = traffic_state.get_state()
        reward = calculate_reward(state, action, next_state)
        done = True if time_step == stop_time - 1 else False

        agent.remember(state, action, reward, next_state, done)

        train_counter +=1
        if train_counter % 120 ==0:
            agent.replay(batch_size)
        
        state = next_state

        traffic_state.update()
        traci.simulationStep()

    print("Episode: {}/{}, Score: {}".format(episode, num_episodes, time_step))

agent.save("dqn_model.h5")
traci.close()


    


