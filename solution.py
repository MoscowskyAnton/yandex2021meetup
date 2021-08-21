import numpy as np
#from collections import OrderedDict
#import matplotlib.pyplot as plt
import sys
import time



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
        
    def draw_order(self, order):
        x, y = order.get_start()
        x1, y1 = order.get_final()
        plt.plot(x, y, 'o', color = 'green')
        plt.plot(x1, y1, '*', color = 'green')
        plt.plot([x, x1], [y, y1], '--', color = 'green')
    
    def draw_bot(self, pose, color = 'blue'):
        plt.plot(pose[0], pose[1], '.', color = color)        
        
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
                print("Is it normal?")
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
        del path[0][-1]#?
        del path[1][-1]#?
        
        return path    
            
    def calc_h(self, n1, n2):
        # manhattan metrics
        #return abs(n1.x - n2.x) + abs(n1.y - n2.y)
        # euclidian
        #return np.hypot(n1.x - n2.x, n1.y - n2.y)#*2
        # square
        return (n1.x - n2.x)**2 + (n1.y - n2.y)**2
        
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
        
        self.path = None
        
    def __str__(self):
        return "[t={}] ({};{})-->({};{})".format(self.appear_t, self.s_row, self.s_col, self.f_row, self.f_col)
    
    def __repr__(self):
        return self.__str__()
    
    def get_start(self):
        return (self.s_row, self.s_col)
    
    def get_final(self):
        return (self.f_row, self.f_col)

class InputReader(object):
    def __init__(self):
        pass
    
    def read_params1(self):
        #params = sys.stdin.readline().split()
        params = list(map(int, input().split(' ')))
        self.N = params[0] # map size
        self.MaxTips = params[1] # reward for order
        self.Cost_c = params[2] # rover cost
        
    def read_map(self):
        self.MAP = Map(self.N)
        for i in range(self.N):
            map_line = input()#sys.stdin.readline()
            for j in range(self.N):# must be
                if map_line[j] == '#':
                    self.MAP.add_obst(i,j)
                if map_line[j] == '.':
                    self.MAP.add_free(i,j)
                    
    def read_params2(self):
        params = list(map(int, input().split(' ')))
        self.T = params[0] # iterations
        self.D = params[1] # total orders number
        
    def read_iteration(self, num):
        iteration = []
        iter_order_num = int(sys.stdin.readline())
        
        for j in range(iter_order_num):
            #iteration.append(Order(sys.stdin.readline().split(), num))
            iteration.append(Order(list(map(int, input().split(' '))), num))
        return iteration

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
        
        self.iterations = {}#OrderedDict()
        for i in range(self.T):
            self.iterations[i] = []
            iter_order_num = int(file.readline())
            for j in range(iter_order_num):
                self.iterations[i].append(Order(file.readline().split(), i))
                
    def printt(self):
        print("N = {}".format(self.N))
        print("MaxTips = {}".format(self.MaxTips))
        print("Cost_c = {}".format(self.Cost_c))
        print("Map:\n{}".format(self.MAP.MAP))
        print("Iterations:\n{}".format(self.iterations))

class Robot(object):
    IDLE = 0
    REACHING_ORDER = 1
    PICKING_ORDER = 2
    DELIVERING = 3
    UNLOADING = 4
    
    def __init__(self, x, y, id):
    
        self.id = id        
        self.x = x
        self.y = y
        
        self.state = Robot.IDLE
        
        self.path1 = []
        self.path2 = []
        
        self.ord_time = None
        self.steps = 0
        
        
    def get_pose(self):
        return (self.x, self.y)
    
    def __str__(self):        
        str_path = 'no path'
        if len(self.path1) and len(self.path2):
            if len(self.path1[0]) and len(self.path2[0]):
                str_path = "({};{})->({};{})".format(self.path1[0][-1], self.path1[1][-1], self.path2[0][0], self.path2[1][0])
        return "[BOT{} S:{}|P:({};{})|{}]".format(self.id, self.state, self.x, self.y, str_path)
    
    def __repr__(self):
        return self.__str__()
        
class OrderTable(object):
    def __init__(self, robots, MAP, MaxTips, Cost_c):
        self.MAP = MAP        
        self.free_robots = robots                
        self.busy_robots = []                
        self.orders = []        
        self.total_time = 0
        self.MaxTips = MaxTips
        self.RobotsCost = Cost_c * len(robots)
        
        self.reward = 0
        
    def add_iter(self, iteration):        
        #for order in iteration:
            #order.path = self.MAP.plan(order.get_start(), order.get_final())
            
        self.orders += iteration
            
    def make_table(self):        
        '''
        self.full_table = []
        for bi in range(len(self.free_robots)):
            self.full_table.append([])
            for oi in range(len(self.orders)):
                self.full_table[bi].append(None)
        '''
        self.score_table = np.zeros((len(self.free_robots),len(self.orders)))        
        
        for oi, order in enumerate(self.orders):
            for bi, bot in enumerate(self.free_robots):                
                #self.full_table[bi][oi] = (self.MAP.plan(bot.get_pose(),order.get_start()), order.path)
                
                #self.score_table[bi,oi] = len(self.full_table[bi][oi][0][0])+len(self.full_table[bi][oi][1][0]) + (self.total_time - order.appear_t* 60) 

                self.score_table[bi,oi] = np.hypot(bot.get_pose()[0]-order.get_start()[0],bot.get_pose()[1]-order.get_start()[1]) + np.hypot(order.get_start()[0]-order.get_final()[0],order.get_start()[1]-order.get_final()[1]) + (self.total_time - order.appear_t* 60) 
                
                # TODO expiration time and if more than MaxTips dont do that        
        #print(self.score_table)
        #print("==================")
        #print(self.full_table) # same! wtf
    
    def greedy_task(self):
        while len(self.free_robots) and len(self.orders):
            #print(self.score_table)            
            r, o = np.unravel_index(np.argmin(self.score_table), self.score_table.shape)
            #print("id{}(#{})->{}".format(self.free_robots[r].id,r, o))
            #print(r,o, self.free_robots)
            #print(self.full_table) # here full table same
            
            # those indexes can be wrong, but sometimes
            current = self.free_robots[r]
            current.path1 = self.MAP.plan(self.free_robots[r].get_pose(), self.orders[o].get_start())#self.full_table[r][o][0]
            current.path2 = self.MAP.plan(self.orders[o].get_start(), self.orders[o].get_final())#self.full_table[r][o][1]
            current.state = Robot.REACHING_ORDER
            current.ord_time = self.orders[o].appear_t * 60 
            current.steps = 0
            
            self.busy_robots.append(current)
            del self.free_robots[r]
                                    
            del self.orders[o]
            
            self.score_table = np.delete(self.score_table, r, 0)
            self.score_table = np.delete(self.score_table, o, 1)
            #print(self.score_table)  
            #print(self.full_table)
            #print('len',len(self.full_table))
            
            #for row in self.full_table:
                #print('row len', len(row))
            
            #del self.full_table[r]
            #for row in self.full_table:
                ##print('ln ind row',len(row), o, row)
                #del row[o]
            
            
    def print_bots(self):
        print('free',self.free_robots)
        print('busy',self.busy_robots)
        
    def do_iter(self, draw = False):
        self.rover_actions = []
        for r in range(len(self.free_robots)+len(self.busy_robots)):
            self.rover_actions.append([])
        
        for s in range(60):
            self.make_table()                            
            self.greedy_task()                
            self.do_step(draw) 
    
    def motion(self, start, end):
        dx = start[0] - end[0]
        dy = start[1] - end[1]
        if dy == 1:
            return 'L'
        if dy == -1:
            return 'R'
        if dx == 1:
            return 'U'
        if dx == -1:
            return 'D'
        if dx == 0 and dy == 0:
            return 'S'
        return '#'
    
    def do_step(self, draw = False):
        
        #self.print_bots()
        remove_idle = []
        for bi, bot in enumerate(self.busy_robots):
            bot.steps+=1
            if bot.state == Robot.REACHING_ORDER:
                if len(bot.path1[0]) == 0:
                    bot.state = Robot.PICKING_ORDER
                    self.rover_actions[bot.id].append('T')
                else:
                    self.rover_actions[bot.id].append(self.motion(bot.get_pose(), (bot.path1[0][-1], bot.path1[1][-1])))
                    bot.x = bot.path1[0][-1]
                    bot.y = bot.path1[1][-1]
                    del bot.path1[0][-1]
                    del bot.path1[1][-1]
            elif bot.state == Robot.PICKING_ORDER or bot.state == Robot.DELIVERING:
                if bot.state == Robot.PICKING_ORDER:
                    bot.state = Robot.DELIVERING
                if len(bot.path2[0]) == 0:
                    bot.state = Robot.UNLOADING
                    self.rover_actions[bot.id].append('P')
                else:
                    self.rover_actions[bot.id].append(self.motion(bot.get_pose(), (bot.path2[0][-1], bot.path2[1][-1])))
                    bot.x = bot.path2[0][-1]
                    bot.y = bot.path2[1][-1]
                    del bot.path2[0][-1]
                    del bot.path2[1][-1]
            elif bot.state == Robot.UNLOADING:
                bot.state = Robot.IDLE
                self.rover_actions[bot.id].append('S')
                self.reward += max(0, self.MaxTips - (self.total_time - bot.ord_time))
            elif bot.state == Robot.IDLE:
                remove_idle.append(bi)
                bot.path1 = []
                bot.path2 = []
                #self.rover_actions[bot.id].append('S')
                
        #self.print_bots()
        for bi in remove_idle:
            self.free_robots.append(self.busy_robots[bi])            
        for bi in sorted(remove_idle, reverse=True):
            del self.busy_robots[bi]
            
        for bi, bot in enumerate(self.free_robots):
            self.rover_actions[bot.id].append('S')
        
        if draw:
            for bot in self.free_robots:
                self.MAP.draw_bot(bot.get_pose(), 'blue')
            for bot in self.busy_robots:
                self.MAP.draw_bot(bot.get_pose(), 'red')
            plt.title(f"Time {self.total_time} Reward {self.reward} Bots cost {self.RobotsCost}")
            plt.pause(0.0001)
        
        self.total_time += 1
               
    
def test_on_file(stop_time_kostyl = {},robot_num_kostyl = {}, draw = False, test_num = 1):    
    tock = time.clock()
    test_file_path = f'test_data/0{test_num}'
    IE = InputEmulator(test_file_path)
    #IE.print()
    
    if IE.N in stop_time_kostyl:
        stop_time = stop_time_kostyl[IE.N]
    else:
        stop_time = float('inf')
    
    R = 1
    if IE.N in robot_num_kostyl:
        R = robot_num_kostyl[IE.N]
        
    rovers = []
    for r in range(R):        
        rover_start = IE.MAP.get_random_free_point()
        rovers.append(Robot(rover_start[0], rover_start[1], r))
        #rovers.append(Robot(3, 3, r))    
        
    OT = OrderTable(rovers, IE.MAP, IE.MaxTips, IE.Cost_c)
    
    if draw:
        OT.MAP.draw_map()
        for bot in rovers:
            OT.MAP.draw_bot(bot.get_pose(), 'yellow')
    #OT.print_bots()        
    
    for i in range(IE.T):
        if time.clock() - tock < stop_time:
            #print("{}/{}".format(i,IE.T))
            if draw:
                for order in IE.iterations[i]:
                    OT.MAP.draw_order(order)      
                    
            OT.add_iter(IE.iterations[i])                        
            OT.do_iter()   
            for bot_a in OT.rover_actions:
                action = "".join(bot_a)
                #sys.stdout.write(f"{action}\n")
        else:
            OT.orders = []
            OT.do_iter() 
            for bot_a in OT.rover_actions:
                action = "".join(bot_a)
                #sys.stdout.write(f"{action}\n")
            #for r in range(R):
            #    print('S'*60)
            
    tick = time.clock()
    print(f"Time = {tick-tock}")
    print(f"Reward {OT.reward} Cost {OT.RobotsCost}")
    

def work_with_input(stop_time_kostyl = {}, robot_num_kostyl = {}):
    tock = time.clock()
    IR = InputReader()
    IR.read_params1()
    IR.read_map()
    #IR.MAP.MAP = IR.MAP.MAP.T
    IR.read_params2()
    
    R = 1
    if IR.N in robot_num_kostyl:
        R = robot_num_kostyl[IR.N]
    
    if IR.N in stop_time_kostyl:
        stop_time = stop_time_kostyl[IR.N]
    else:
        stop_time = float('inf')
    sys.stdout.write(f"{R}\n")
    sys.stdout.flush()
    
    
    rovers = []
    rovers_str = ""
    for r in range(R):        
        rover_start = IR.MAP.get_random_free_point()
        #rover_start = (3,3)
        rovers.append(Robot(rover_start[0], rover_start[1], r))
        sys.stdout.write(f"{rover_start[0]+1} {rover_start[1]+1}\n")
        sys.stdout.flush()
        
    OT = OrderTable(rovers, IR.MAP, IR.MaxTips, IR.Cost_c)
    for i in range(IR.T):
        iteration = IR.read_iteration(i)
        if time.clock() - tock < stop_time:
            OT.add_iter(iteration)
            OT.do_iter()
            for bot_a in OT.rover_actions:
                action = "".join(bot_a)
                sys.stdout.write(f"{action}\n")
                sys.stdout.flush()
        else:
            OT.orders = []
            OT.do_iter()
            for bot_a in OT.rover_actions:
                action = "".join(bot_a)
                sys.stdout.write(f"{action}\n")
                sys.stdout.flush()
    
if __name__ == '__main__': 
    stop_time_kostyl = {4:19, 128:16, 180:16, 384:4, 1024:3, 1000:3}
    robot_num_kostyl = {4: 1, 128:2, 180:2, 384:1, 1024:1, 1000:1}
    #for i in range(1,10):
        #print(f"Test {i}")
        #test_on_file(stop_time_kostyl, robot_num_kostyl, False, i)
    work_with_input(stop_time_kostyl, robot_num_kostyl)   
    
