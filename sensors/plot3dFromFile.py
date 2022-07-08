import matplotlib.pyplot as plt

filePath = "/home/donkarlo/mrs_workspace/src/trajectory_loader/sample_trajectories/circle_10_5.txt"
# Using readlines()
file1 = open(filePath, 'r')
Lines = file1.readlines()

count = 0
# Strips the newline character
x = []
y = []
for line in Lines:
    splitted = line.split(',')
    x.append(float(splitted[0]))
    y.append(float(splitted[1]))

plt.plot(x, y)
plt.show()
