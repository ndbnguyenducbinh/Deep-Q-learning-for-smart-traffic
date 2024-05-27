import os
import traci
import sys

# Assume SUMO_HOME is set in your environment variables
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoCmd = ["sumo-gui", "-c", "/media/binhnguyenduc/ubuntu_data/vscode/Simulate_sumo/Env/map_tl_2.sumocfg"]

# Start TraCI
traci.start(sumoCmd)

# List all traffic light IDs
tl_ids = traci.trafficlight.getIDList()
print("Traffic Light IDs:", tl_ids)

# Fetch and print details for each traffic light
for tl_id in tl_ids:
    print(f"Traffic Light {tl_id} Program Logic:")
    try:
        logics = traci.trafficlight.getAllProgramLogics(tl_id)
        print(logics)

        for logic in logics:
            print(f"  Logic ID: {logic.programID}")
            for phase in logic.phases:
                print(f"    Duration: {phase.duration}, State: {phase.state}")
    except Exception as e:
        print("Error fetching data for", tl_id, ":", e)

# Stop TraCI
traci.close()
'''
(Logic(programID='0', type=3, currentPhaseIndex=0, phases=(Phase(duration=42.0, state='GggggGrrrrGggggGrrrr', minDur=42.0, maxDur=42.0), 
                                                           Phase(duration=3.0, state='GyyyyGrrrrGyyyyGrrrr', minDur=3.0, maxDur=3.0), 
                                                           Phase(duration=42.0, state='GrrrrGggggGrrrrGgggg', minDur=42.0, maxDur=42.0), 
                                                           Phase(duration=3.0, state='GrrrrGyyyyGrrrrGyyyy', minDur=3.0, maxDur=3.0)), subParameter={}),)
'''