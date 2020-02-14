import argparse
import time
import msgpack
import matplotlib.pyplot as plt

from enum import Enum, auto
import numpy as np
import re

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local, local_to_global

from probabilistic_roadmap import create_graph_with_nodes, a_star, heuristic


class MapData:
    def __init__(self, map_file_name):
        print("Loading grid from file ...")
        self.TARGET_ALTITUDE = 5
        self.SAFETY_DISTANCE = 5
        self.waypoints = []

        # TODO: read lat0, lon0 from colliders into floating point values
        header = open(map_file_name).readline()
        s = re.findall(r"[-+]?\d*\.\d+|\d+", header)
        self.lat_0 = float(s[1])
        self.lon_0 = float(s[3])

        grid_home_lat0 = 316
        grid_home_lon0 = 445 
        
        # Read in obstacle map
        self.data = np.loadtxt(map_file_name, delimiter=',', dtype='Float64', skiprows=2)
        self.grid, self.graph, self.grid_start, self.grid_goal = \
        create_graph_with_nodes(self.data, grid_home_lat0, grid_home_lon0, self.TARGET_ALTITUDE)
        

    def search_path(self):
        print("Searching for a path ...")

        # plot grid with start and goal point
        fig = plt.figure(figsize=(5,6)) 
        plt.imshow(self.grid, origin='lower', cmap='Greys')

        print('Local Start and Goal: ', self.grid_start, self.grid_goal)


        path, cost = a_star(self.graph, heuristic, self.grid_start, self.grid_goal)
        if len(path) > 0:
            print('Found a Path len({0}): {1}'.format(len(path),path))
        else:
            print('*** No Path Found ***')

        path_pairs = zip(path[:-1], path[1:])
        for (n1, n2) in path_pairs:
            print(n1, n2)
        
        nmin = np.min(self.data[:, 0])
        emin = np.min(self.data[:, 1])
            
        # draw nodes
        print('draw nodes')
        for n1 in self.graph.nodes:
            plt.scatter(int(n1[1] - emin), int(n1[0] - nmin), c='red')
            
        # draw edges
        print('draw edges')
        for (n1, n2) in self.graph.edges:
            plt.plot([int(n1[1] - emin), int(n2[1] - emin)], [int(n1[0] - nmin), int(n2[0] - nmin)], 'black')
            
        # TODO: add code to visualize the path
        print('draw path')
        path_pairs = zip(path[:-1], path[1:])
        for (n1, n2) in path_pairs:
            plt.plot([int(n1[1] - emin), int(n2[1] - emin)], [int(n1[0] - nmin), int(n2[0] - nmin)], 'green')

        # draw start, goal
        plt.plot(int(self.grid_start[1] - emin), int(self.grid_start[0]  - nmin), 'bo')
        plt.plot(int(self.grid_goal[1]  - emin), int(self.grid_goal[0]  - nmin), 'bx')
        fig.canvas.draw()
        plt.pause(0.1)

        def onclick(event):
            # Convert path to waypoints
            waypoints = [[int(p[0]), int(p[1]), int(p[2]), 0] for p in path]
            # Set self.waypoints
            self.waypoints = waypoints
            print('waypoints:{0}'.format(waypoints))
            plt.close(1)
                    
            return
        # select start and destination
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.show()


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection, m):
        print('____Init____')
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = m.waypoints
        self.in_mission = True
        self.check_state = {}
        self.lat_0 = m.lat_0
        self.lon_0 = m.lon_0
        self.TARGET_ALTITUDE = m.TARGET_ALTITUDE

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)


    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                print('>>> Manual->Armed')
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                print('>>> Arming->takeoff')
                self.takeoff_transition()
            elif self.flight_state == States.TAKEOFF:
                print('>>> Takeoff->Waypoint')
                print('ARM Waypoint count:{0}'.format(len(self.waypoints)))
                print('ARMED:{0}'.format(self.armed))
                if self.armed and len(self.waypoints) > 0:
                    self.send_waypoints()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                   self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("<arming transition>")
        self.arm()
        self.take_control()
        self.set_home_position(self.lon_0, self.lat_0, 0)
        print(f'Home lat : {self.lat_0}, lon : {self.lon_0}')
        print('Armed:{0}'.format(self.armed))

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("<takeoff transition>")
        self.target_position[2] = self.TARGET_ALTITUDE
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("<waypoint transition>")
        if len(self.waypoints) > 0:
            self.target_position = self.waypoints.pop(0)
            print('target position', self.target_position)
            self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("<landing transition>")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("<disarm transition>")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("<manual transition>")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        self.flight_state = States.WAYPOINT
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)


    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=6000)
    m = MapData('colliders.csv')
    m.search_path()

    if len(m.waypoints) > 0:
        drone = MotionPlanning(conn, m=m)
        #time.sleep(1)
        print('******START********')
        drone.start()