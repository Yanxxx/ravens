import numpy as np
import pybullet as p
import pybullet_data
# from ravens.utils import p_utils

p.connect(p.GUI)
p.resetSimulation()

plane = p.loadURDF("/home/yan/git/ravens/ravens/environments/assets/plane/plane.urdf")
# robot = p_utils.load_urdf('ur5.urdf')
robot = p.loadURDF("ur5.urdf", [0, 0, 0], useFixedBase=1)

position, orientation = p.getBasePositionAndOrientation(robot)

print(p.getNumJoints(robot))

#joint_positions = [j[0] for j in p.getJointStates(robot, range(6))]
joint_positions = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
targetPositions = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
world_position, world_orientation = p.getLinkState(robot, 2)[:2]
print(world_position)

p.setGravity(0, 0, -9.81)   # everything should fall down
p.setTimeStep(0.0001)       # this slows everything down, but let's be accurate...
p.setRealTimeSimulation(0)  # we want to be faster than real time :)


p.setJointMotorControlArray(
    robot, range(6), p.POSITION_CONTROL,
    targetPositions)

print('init position')
world_position, world_orientation = p.getLinkState(robot, 8)[:2]
print(world_position)

p.setGravity(0, 0, -9.81)
p.setTimeStep(0.0001)
p.setRealTimeSimulation(0)


p.setJointMotorControlArray(
    robot, range(6), p.POSITION_CONTROL,
    targetPositions)
for _ in range(10000):
    p.stepSimulation()
    

for i in range(p.getNumJoints(robot)):
    joint_index = i
    joint_info = p.getJointInfo(robot, joint_index)
    name, joint_type, lower_limit, upper_limit = \
        joint_info[1], joint_info[2], joint_info[8], joint_info[9]
    print(name, joint_type, lower_limit, upper_limit)
    print('init position')
    world_position, world_orientation = p.getLinkState(robot, i)[:2]
    print(world_position)
print("Hello!")
input("Press the <Enter> key on the keyboard to exit.")
