#!/usr/bin/env python3
import rospy
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import AttitudeTarget
import math
import time
from tensorflow.keras import layers
from mavros_msgs.msg import PositionTarget
from stalker.msg import PREDdata
from BoxToCenter import center_detector
import csv
from collections import deque
import time
import os
#-------------------------------- NOISE CLASS --------------------------------#
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-1, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

#-------------------------------- CLASS BUFFER --------------------------------#

class Buffer:
    def __init__(self, buffer_capacity = 100000, batch_size = 64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
    def record(self, obs_tuple):  
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1  

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True) 
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))  

#-------------------------------- CLASS ENVIRONMENT --------------------------------#
class Environment:
    def __init__(self):
        self.pub_pos = rospy.Publisher("/mavros/setpoint_raw/local",PositionTarget,queue_size=10000)
        self.pub_action = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10000)
        self.initial_pose()
        self.x_initial = 0.0
        self.y_initial = 0.0
        self.z_initial = 4.0
        self.yaw_initial = 90.0
        self.x_initial_noise = np.random.uniform(-1, 1)
        self.y_initial_noise = np.random.uniform(-1, 1)
        self.x_position = 0.0
        self.y_position = 0.0
        self.z_position= 4.0
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.z_velocity = 0.0 
        self.x_angular = 0.0
        self.y_angular = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 90.0
        self.action_history = deque(maxlen=5)
        self.consecutive_episodes = 0
        self.exceeded_bounds = False
        self.to_start = False
        self.timestep = 1
        self.current_episode = 1
        self.episodic_reward = 0.0
        self.previous_state = np.zeros(num_states)
        self.action = np.zeros(num_actions)
        self.previous_action = np.zeros(num_actions)
        self.done = False
        self.max_timesteps = 1024
        self.ngraph = 0
        self.max_avg_reward = -1000
        self.sub_detector = rospy.Subscriber("/box", PREDdata, self.DetectCallback)
        self.sub_position = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.PoseCallback)
        self.sub_target = rospy.Subscriber("/robot/robotnik_base_control/odom", Odometry, self.TargetCallback)
        self.box = PREDdata()
        self.desired_pos_z = 4.0
        self.new_pose = False
        self.detector = center_detector()
        self.distance_x = 0
        self.distance_y = 0
        self.angle = 0
        self.ddist_x = 0
        self.ddist_y = 0
        self.dt = 0
        self.time_prev = 0
        self.current_stage = 1  # 课程学习阶段

        self.stage_reward_multiplier = [1, 1.5, 2]  # 随阶段增加奖励
        self.timestamp = time.strftime("%m%d_%H%M%S")
        absolute_path=os.path.abspath(__file__)
        current_directory=os.path.dirname(absolute_path)
        self.filename='/home/lxy/zjy/catkin_ws/src/stalker/scripts/checkpoints/follow'+absolute_path[len(current_directory):len(absolute_path)-3]+'/'
        print(self.filename)
        os.makedirs(self.filename+str(self.timestamp),exist_ok=True)
        with open(self.filename+str(self.timestamp)+'/training_error.csv','x')as f:
            1
            pass
        with open(self.filename+str(self.timestamp)+'/training_reward.csv', 'x') as f:
            1
            pass
    def initial_pose(self):
        action_mavros = AttitudeTarget()
        action_mavros.type_mask = 7
        action_mavros.thrust = 0.5 # Altitude hold
        action_mavros.orientation = self.rpy2quat(0.0,0.0,90.0) # 90 yaw
        self.pub_action.publish(action_mavros)
    def rpy2quat(self,roll,pitch,yaw):
        
        q = Quaternion()
        r = np.deg2rad(roll)
        p = np.deg2rad(pitch)
        y = np.deg2rad(yaw)
        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cp = math.cos(p * 0.5)
        sp = math.sin(p * 0.5)
        cr = math.cos(r * 0.5)
        sr = math.sin(r * 0.5)
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q   
    def quat2rpy(self,quat):
        sinr_cosp = 2.0*(quat.w*quat.x + quat.y*quat.z)
        cosr_cosp = 1 - 2*(quat.x*quat.x + quat.y*quat.y)
        roll = math.atan2(sinr_cosp , cosr_cosp)    
        sinp = 2*(quat.w*quat.y - quat.z*quat.x)
        if abs(sinp)>=1:
            pitch = math.pi/2.0 * sinp/abs(sinp)
        else:
            pitch = math.asin(sinp)
        siny_cosp = 2*(quat.w*quat.z + quat.x*quat.y)
        cosy_cosp = 1 - 2*(quat.y*quat.y + quat.z*quat.z)
        yaw = math.atan2(siny_cosp,cosy_cosp)
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)  
        return roll, pitch, yaw      
    def go_to_start(self):
        position_reset = PositionTarget()
        position_reset.type_mask = 2496
        position_reset.coordinate_frame = 1
        position_reset.position.x = self.x_initial + self.x_initial_noise
        position_reset.position.y = self.y_initial + self.y_initial_noise
        position_reset.position.z = self.z_initial
        position_reset.yaw = self.yaw_initial
        self.pub_pos.publish(position_reset) 

    def reset(self):
        ep_reward_list.append(self.episodic_reward*self.max_timesteps/self.timestep)
        avg_reward = np.mean(ep_reward_list[-40:])
        episodes.append(self.current_episode)
        self.check_stage_transition()
        print(f"当前课程阶段: {self.current_stage}")
        print("timesteps :", self.timestep)
        print("Episode * {} * Cur Reward is ==> {}".format(self.current_episode,self.episodic_reward*self.max_timesteps/self.timestep))
        print("Episode * {} * Avg Reward is ==> {}".format(self.current_episode, avg_reward))
        avg_reward_list.append(avg_reward)
        with open(self.filename+str(self.timestamp)+'/training_reward.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            data = [ avg_reward, self.episodic_reward*self.max_timesteps/self.timestep ]
            writer.writerow(data)
        #if (avg_reward != 0):
        if (avg_reward > self.max_avg_reward and avg_reward != 0):
            self.max_avg_reward = avg_reward
            actor_model.save_weights(self.filename+str(self.timestamp)+"/ddpg_actor"+str(self.ngraph)+".h5")
            critic_model.save_weights(self.filename+str(self.timestamp)+"/ddpg_critic"+str(self.ngraph)+".h5")
            target_actor.save_weights(self.filename+str(self.timestamp)+"/ddpg_target_actor"+str(self.ngraph)+".h5")
            target_critic.save_weights(self.filename+str(self.timestamp)+"/ddpg_target_critic"+str(self.ngraph)+".h5")     
            print("-----Weights saved-----")

        plt.figure(0) 
        plt.title('training reward', fontsize=10)
        plt.plot(ep_reward_list, 'b', label='ep_reward')
        plt.plot(avg_reward_list, 'r', label='avg_reward')
        plt.ylabel('Score')
        plt.xlabel('Episodes')
        plt.legend()
        plt.grid()
        plt.savefig(self.filename+str(self.timestamp)+'/ddpg_score'+str(self.ngraph)+'.eps', format='eps')
        plt.clf()
        plt.figure(1)
        plt.title('distance and angle error', fontsize=10)
        plt.plot(angles, 'g', label='angle')
        plt.plot(distances_x, 'b',label='distance_x')
        plt.plot(distances_y, 'r',label='distance_y')
        plt.grid()
        plt.legend()
        plt.savefig(self.filename+str(self.timestamp)+'/distancexy-angle'+str(self.ngraph)+'.eps', format='eps')
        plt.clf()
        plt.figure(2) 
        plt.title('training reward', fontsize=10)
        plt.plot(ep_reward_list, 'b', label='ep_reward')
        plt.plot(avg_reward_list, 'r', label='avg_reward')
        plt.ylabel('Score')
        plt.xlabel('Episodes')
        plt.legend()
        plt.grid()
        plt.savefig(self.filename+str(self.timestamp)+'/ddpg_score'+str(self.ngraph))
        plt.clf()
        plt.figure(3)
        plt.title('distance and angle error', fontsize=10)
        plt.plot(angles, 'g', label='angle')
        plt.plot(distances_x, 'b',label='distance_x')
        plt.plot(distances_y, 'r',label='distance_y')
        plt.grid()
        plt.legend()
        plt.savefig(self.filename+str(self.timestamp)+'/distancexy-angle'+str(self.ngraph))
        plt.clf()   
        print("-----Plots saved-----")



        if self.current_episode % 200 == 0.0:
            self.ngraph += 1
            #we do this so we reduce memory used and take less time to save the graphs (less delay in training)
            self.max_avg_reward = -1000 #reset for every 200 episodes, we get the max weights in each graph
        self.episodic_reward = 0.0
        self.current_episode += 1
        self.timestep = 1
        self.done = False
        self.exceeded_bounds = False  
        self.to_start  = False 
        self.x_initial_noise = np.random.uniform(-1, 1)
        self.y_initial_noise = np.random.uniform(-1, 1)

    # 判断阶段过渡的条件
    def check_stage_transition(self):
        
        if self.current_stage == 1:
            # 当前时间步是否大于900的判断
            if self.timestep ==1025:
                self.consecutive_episodes += 1  # 连续回合数+1
            else:
                self.consecutive_episodes = 0  # 不满足条件时重置

            # 如果连续满足二十个回合，进入第二阶段
            if self.consecutive_episodes >= 20:
                print("进入第二阶段")
                self.current_stage = 2
                self.consecutive_episodes = 0  # 重置计数器
                
    def PoseCallback(self,msg):
        self.position = msg
        self.x_position = self.position.pose.pose.position.x
        self.y_position = self.position.pose.pose.position.y
        self.z_position = self.position.pose.pose.position.z

        self.x_velocity = self.position.twist.twist.linear.x 
        self.y_velocity = self.position.twist.twist.linear.y
        self.z_velocity = self.position.twist.twist.linear.z 

        self.x_angular = self.position.twist.twist.angular.x
        self.y_angular = self.position.twist.twist.angular.y

        quat = self.position.pose.pose.orientation
        self.roll, self.pitch, self.yaw = self.quat2rpy(quat)
        self.new_pose = True

    def TargetCallback(self,msg):
        self.x_initial = (-1.0)*msg.pose.pose.position.y  
        self.y_initial = msg.pose.pose.position.x + 2.0   #initial relative position of drone, odom doesnt use global coordinates!

    def DetectCallback(self, msg):
        if self.new_pose == False:
            return
        else:
            self.new_pose = False
            self.box = msg
            self.distance_x, self.distance_y, self.angle = self.detector.compute(self.box, self.roll, self.pitch, self.z_position)
            rostime_now = rospy.get_rostime()
            self.time_now = rostime_now.to_nsec()
            if self.time_prev == 0:
                self.dt = 0
            else:
                self.dt = (self.time_now - self.time_prev)/1e9
            self.time_prev = self.time_now
            if self.distance_x == 10000 and self.distance_y == 10000 :
                self.exceeded_bounds = True
            if self.exceeded_bounds and not self.done : # Bounds around desired position
                print("Exceeded Bounds --> Return to initial position")
                self.done = True 
            elif self.timestep > self.max_timesteps and not self.done:
                print("Reached max number of timesteps --> Return to initial position") 
                self.reward += 200  
                self.reset()                 
                print("Reset")                   
                print("Begin Episode %d" %self.current_episode)
                #we miss 2 timesteps between episodes while bot is still moving
       
            if self.done:
                self.go_to_start()
                if abs(self.x_position-self.x_initial-self.x_initial_noise)<0.1 and abs(self.y_position-self.y_initial-self.y_initial_noise)<0.1 and abs(self.z_position-self.z_initial)<0.1 :
                    self.reset()                 
                    print("Reset")                   
                    print("Begin Episode %d" %self.current_episode)      

            else:           
                # Compute the current state
                max_distance_x = 240 #pixels
                max_distance_y = 360
                max_velocity = 2 #m/s
                max_angle = 90 #degrees #bad name of variable ,be careful there is angle_max too for pitch and roll.
                max_derivative = 100
                if self.dt == 0:
                    self.ddist_x = 0
                    self.ddist_y = 0
                else:
                    self.ddist_x = ( self.distance_x - int(self.previous_state[0]*max_distance_x) ) / self.dt
                    self.ddist_y = ( self.distance_y - int(self.previous_state[1]*max_distance_y) ) / self.dt
                    # values -> 2,4,6 pixels (because of resolution reduction in BoxToCenter)
                    # most common 2 pixels movement , /0.1 === *10 => 20 is the most common value 
   
                self.current_state = np.array([self.distance_x/max_distance_x, self.distance_y/max_distance_y, np.clip(self.ddist_x/max_derivative,-1, 1), np.clip(self.ddist_y/max_derivative,-1, 1), np.clip(self.y_velocity/max_velocity, -1, 1), np.clip(self.x_velocity/max_velocity, -1, 1)])
                self.action_history.append(self.current_state)
                if self.timestep > 1:
                    total_reward = 0  # 初始化总奖励


                   # 第一阶段：保持高度和跟踪无人车
                    if self.current_stage == 1:
                        height_error = abs(self.z_position - self.z_initial)
                        #if height_error < 0.5:
                        #    total_reward += 50  # 高度误差小，给予奖励
                        # 位置误差计算
                        distance_error = abs(self.distance_x)/max_distance_x + abs(self.distance_y)/max_distance_y
                        position_error = distance_error
    
                        # 如果误差小于0.1，给予额外奖励
                        if distance_error < 0.1:
                            total_reward += 50  # 增加正向奖励


                        weight_position = 150

                        # 偏航惩罚
                        weight_yaw = 80
                        yaw_pen = abs(self.action[2])/yaw_max
                        if distance_error < 0.5:
                            yaw_pen *= 1 + (distance_error / 0.5)  # 随着靠近目标，增加偏航惩罚

                        # 边缘检测惩罚
                        if abs(self.distance_x) > 0.9 or abs(self.distance_y) > 0.9:
                            total_reward += -100  # 当目标接近边缘时，强烈惩罚
                        # 随机探索奖励
                        #if np.random.rand() < 0.05:
                        #    total_reward += 10  # 随机探索奖励
                        total_reward -= 20 * height_error  # 高度误差惩罚
                        total_reward += -weight_position * position_error
                        total_reward += -weight_yaw * yaw_pen

                    if self.current_stage == 2:
                        height_error = abs(self.z_position - self.z_initial)
                        # 位置误差计算
                        distance_error = abs(self.distance_x)/max_distance_x + abs(self.distance_y)/max_distance_y
                        position_error = distance_error
    
                        # 如果误差小于0.1，给予额外奖励
                        if distance_error < 0.1:
                            total_reward += 50  # 增加正向奖励

                        if distance_error > 0.5:  # 远离目标，优先考虑位置误差
                            weight_position = 100
                            weight_velocity = 30
                            weight_derivative = 30
                        else:  # 接近目标时，注重速度和姿态
                            weight_position = 80
                            weight_velocity = 50
                            weight_derivative = 50

                        # 导数误差和速度误差
                      
                        derivative_error = (abs(self.ddist_x/max_derivative)**2) + (abs(self.ddist_y/max_derivative)**2)
                        velocity_error = (min(abs(self.y_velocity)/max_velocity, 1)**2) + (np.clip((-1)*self.x_velocity/max_velocity, 0, 1)**2)

                        # 大滚转和俯仰角度惩罚
                        action = abs(self.action[0])/angle_max + abs(self.action[1])/angle_max + abs(self.action[2])/yaw_max
                        weight_action = 20

                        # 平稳飞行奖励
                        #if distance_error < 0.1 and abs(self.action[0]) < 0.05 and abs(self.action[1]) < 0.05:
                        #    total_reward += 300  # 对平稳飞行增加奖励
                        # 计算平稳性（判断过去n个时间步的动作变化是否都很小）
                        stable_flight = all(
                abs(a[0]) < 0.05 and abs(a[1]) < 0.05 for a in self.action_history)

                    # 如果距离误差小且过去几个时间步内都保持平稳飞行，增加奖励
                        if distance_error < 0.1 and stable_flight:
                            total_reward += 300  # 平稳飞行的额外奖励

                        # 偏航惩罚
                        weight_yaw = 80
                        yaw_pen = abs(self.action[2])/yaw_max
                        if distance_error < 0.5:
                            yaw_pen *= 1 + (distance_error / 0.5)  # 随着靠近目标，增加偏航惩罚

                        # 边缘检测惩罚
                        if abs(self.distance_x) > 0.9 or abs(self.distance_y) > 0.9:
                            total_reward += -100  # 当目标接近边缘时，强烈惩罚



                        # 累加惩罚项
                        total_reward += -weight_position * position_error
                        total_reward += -weight_derivative * derivative_error
                        total_reward += -weight_velocity * velocity_error
                        total_reward += -weight_action * action
                        total_reward += -weight_yaw * yaw_pen
                        total_reward -= 20 * height_error  # 高度误差惩罚

                    # 更新总奖励（可选择是否归一化）
                    self.reward = total_reward * 0.001 #/ 480  # 根据需求决定是否归一化
          
           
                    buffer.record((self.previous_state, self.action, self.reward, self.current_state ))

                    self.episodic_reward += self.reward
                    buffer.learn()
                    update_target(target_actor.variables, actor_model.variables, tau)
                    update_target(target_critic.variables, critic_model.variables, tau) 

                    distances_x.append(self.distance_x/max_distance_x)
                    distances_y.append(self.distance_y/max_distance_y)
                    angles.append(self.angle/max_angle) 

                    
                self.previous_action = self.action                  

                tf_current_state = tf.expand_dims(tf.convert_to_tensor(self.current_state), 0)
                tf_action = tf.squeeze(actor_model(tf_current_state))
                noise = ou_noise()
                self.action = tf_action.numpy() + noise  # Add exploration strategy
                self.action[0] = np.clip(self.action[0], angle_min, angle_max)
                self.action[1] = np.clip(self.action[1], angle_min, angle_max)
                self.action[2] = np.clip(self.action[2], yaw_min, yaw_max)

                with open(self.filename+str(self.timestamp)+'/training_error.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    data = [ rospy.get_rostime(), self.distance_x/max_distance_x, self.distance_y/max_distance_y, self.angle/max_angle, self.x_velocity, self.y_velocity, self.z_position , self.action[0], self.action[1], self.action[2] ]
                    writer.writerow(data)
                roll_des = self.action[0]
                pitch_des = self.action[1] 
                yaw_des = self.action[2] + self.yaw  #differences in yaw
  
                action_mavros = AttitudeTarget()
                action_mavros.type_mask = 7
                action_mavros.thrust = 0.5
                action_mavros.orientation = self.rpy2quat(roll_des,pitch_des,yaw_des)
                self.pub_action.publish(action_mavros)
                self.previous_state = self.current_state
                self.timestep += 1     
       

#-------------------------------- MAIN --------------------------------#
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau)) 
def get_actor():
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=(num_states,))
    h1 = layers.Dense(256, activation="tanh")(inputs)
    h2 = layers.Dense(256, activation="tanh")(h1)    
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(h2)
    outputs = outputs * [angle_max, angle_max, yaw_max]
        
    model = tf.keras.Model(inputs, outputs)

    return model  

def get_critic():
    state_input = layers.Input(shape=(num_states))
    h1_state = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(h1_state)
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    model = tf.keras.Model([state_input, action_input], outputs)

    return model 

if __name__=='__main__':
    rospy.init_node('rl_node', anonymous=True)
    tf.compat.v1.enable_eager_execution()

    num_actions = 3 
    num_states = 6 
    angle_max = 3.0 
    angle_min = -3.0 # constraints for commanded roll and pitch
    yaw_max = 10.0 #how much yaw should change every time
    yaw_min = -10.0
    checkpoint = 6 #checkpoint try
    ntry = 21
    actor_model = get_actor()

    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()

    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())
    critic_lr = 0.001
    actor_lr = 0.0001
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
    gamma = 0.99
    tau = 0.001   
    ep_reward_list = []
    avg_reward_list = [] 
    episodes = []

    distances_x = []
    distances_y = []
    angles = []
    rewards = []
    Environment()
    buffer = Buffer(100000, 64)
    std_dev = 0.1

    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))

    r = rospy.Rate(20)
    while not rospy.is_shutdown:
        r.sleep()    

    rospy.spin()        

