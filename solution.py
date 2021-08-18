# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

class Map(object):
    def __init__(self, N):
        self.MAP = np.zeros((N, N))
        self.N = N
        self.free_indexes = []
        
    def add_obst(self, i, j):
        self.MAP[i,j] = 1
        
    def add_free(self, i, j):
        self.MAP[i,j] = 0
        # remember free indexes?
        self.free_indexes.append(self.xy_to_index(i, j))
        
    def xy_to_index(self, x, y):
        return y * self.N + x
        
    def index_to_xy(self, index):
        x = index % self.N
        y = int((index - x) / self.N)
        return (x, y)
        
    def draw_map(self):
        plt.pcolor(self.MAP.T, cmap = plt.get_cmap('Greys'))
    
    def draw_path(self, path, color = 'red'):    
        plt.plot(path[0], path[1], '--', color = color)   
        plt.plot(path[0][0], path[1][0], "*", color = color)
    
    def draw_bot(self, pose):
        plt.plot(pose[0], pose[1], 'ob')        
        
    def get_random_free_point(self):
        index = np.random.randint(0, len(self.free_indexes))
        return self.index_to_xy(self.free_indexes[index])
        
        
    ## A*
    class Node(object):
        def __init__(self, x, y, cost, parent_ind):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_ind = parent_ind
            
        def __eq__(self, other):
            return self.x == other.x and self.y == other.y
        
        def add(self, motion_list, pid):
            new_node = Map.Node(self.x + motion_list[0], self.y + motion_list[1], self.cost + motion_list[2], pid)
            return new_node
        
        def __gt__(self, other):
            return self.cost > other.cost
            
    def plan(self, start, goal):
        start_node = self.Node(start[0], start[1], 0.0, -1)
        goal_node = self.Node(goal[0], goal[1], 0.0, -1)
        
        open_nodes = {}
        closed_nodes = {}
        
        open_nodes[self.node_index(start_node)] = start_node
        
        while True:
            # no path?
            if len(open_nodes) == 0:
                break
            
            min_cost_id = min(open_nodes, key=lambda i: open_nodes[i].cost + self.calc_h(open_nodes[i], goal_node))            
            selected_node = open_nodes[min_cost_id]
            
            # goal maybe?
            if selected_node == goal_node:
                goal_node = selected_node # cost and parent_id should be correct
                break
            
            del open_nodes[min_cost_id]            
            closed_nodes[min_cost_id] = selected_node
            
            # move further
            for motion in self.motion():
                new_node = selected_node.add(motion, min_cost_id)
                new_node_id = self.node_index(new_node)
                
                if new_node_id in closed_nodes:
                    continue
                if not self.validate(new_node):
                    continue
                
                if new_node_id not in open_nodes:
                    open_nodes[new_node_id] = new_node
                else:
                    if open_nodes[new_node_id] > new_node:
                        open_nodes[new_node_id] = new_node # for better path
                        
        path = [[],[]]
        path[0].append(goal_node.x)
        path[1].append(goal_node.y)
        parent_ind = goal_node.parent_ind
        while parent_ind != -1:
            node = closed_nodes[parent_ind]
            path[0].append(node.x)
            path[1].append(node.y)
            parent_ind = node.parent_ind
            
        return path
            
            
            
    def calc_h(self, n1, n2):
        # manhattan metrics
        #return abs(n1.x - n2.x) + abs(n1.y - n2.y)
        # euclidian
        return np.hypot(n1.x - n2.x, n1.y - n2.y)
        
    def node_index(self, node):
        return self.xy_to_index(node.x, node.y)
                    
    def motion(self):
        return [[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]]
    
    def validate(self, node):
        # borders
        if node.x >= self.N or node.y >= self.N or node.x < 0 or node.y < 0:
            return False        
        # obstacles
        if self.MAP[node.x, node.y] == 1:
            return False
        return True

class Order(object):
    def __init__(self, string, appear_t):
        self.s_row = int(string[0])-1
        self.s_col = int(string[1])-1
        self.f_row = int(string[2])-1
        self.f_col = int(string[3])-1
        
        self.appear_t = appear_t
        
    def __str__(self):
        return "[t={}] ({};{})-->({};{})".format(self.appear_t, self.s_row, self.s_col, self.f_row, self.f_col)
    
    def __repr__(self):
        return self.__str__()

class InputEmulator(object):
    def __init__(self, file_path):
                        
        file = open(file_path, 'r')
        params = file.readline().split()
        self.N = int(params[0]) # map size
        self.MaxTips = int(params[1]) # reward for order
        self.Cost_c = int(params[2]) # rover cost
        
        self.MAP = Map(self.N)
        for i in range(self.N):
            map_line = file.readline()
            for j in range(self.N):# must be
                if map_line[j] == '#':
                    self.MAP.add_obst(i,j)
                if map_line[j] == '.':
                    self.MAP.add_free(i,j)
                    
        params = file.readline().split()
        self.T = int(params[0]) # iterations
        self.D = int(params[1]) # total orders number
        
        self.iterations = OrderedDict()
        for i in range(self.T):
            self.iterations[i] = []
            iter_order_num = int(file.readline())
            for j in range(iter_order_num):
                self.iterations[i].append(Order(file.readline().split(), i))
                
    def print(self):
        print("N = {}".format(self.N))
        print("MaxTips = {}".format(self.MaxTips))
        print("Cost_c = {}".format(self.Cost_c))
        print("Map:\n{}".format(self.MAP.MAP))
        print("Iterations:\n{}".format(self.iterations))
                                   
            
if __name__ == '__main__':
    #print(np.__version__)
    
    test_file_path = 'test_data/08'
    IE = InputEmulator(test_file_path)
    #IE.print()
    
    rover_start = IE.MAP.get_random_free_point()
    for i in range(IE.T):
        if len(IE.iterations[i]) != 0:
            plan1 = IE.MAP.plan((IE.iterations[i][0].s_row, IE.iterations[i][0].s_col), (IE.iterations[i][0].f_row, IE.iterations[i][0].f_col))
            plan2 = IE.MAP.plan(rover_start, (IE.iterations[i][0].s_row, IE.iterations[i][0].s_col))
            break
    #print(plan, len(plan))
    
    IE.MAP.draw_map()
    IE.MAP.draw_path(plan1, 'red')
    IE.MAP.draw_path(plan2, 'blue')
    IE.MAP.draw_bot(rover_start)
    print(IE.MaxTips - len(plan1[0]) + len(plan2[0]))
    plt.show()
    
    '''
    node = IE.MAP.Node(2,0,0,-1)
    i = IE.MAP.node_index(node)
    print(i)
    x,y = IE.MAP.node_xy(node, i)
    print(x,y)
    '''