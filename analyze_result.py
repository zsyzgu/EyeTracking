import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    lines = open('result.txt', 'r').readlines()

    pc_x = {}
    pc_y = {}

    for line in lines:
        tags = line.strip().split()
        label = '_'.join(tags[0].split('_')[0:-1])
        x = np.mean([float(tags[1]), float(tags[3])])
        y = np.mean([float(tags[2]), float(tags[4])])
        if label not in pc_x.keys():
            pc_x[label] = []
            pc_y[label] = []
        pc_x[label].append(x)
        pc_y[label].append(y)
    
    for key in pc_x.keys():
        plt.scatter(pc_x[key], pc_y[key])
    plt.show()

    for line in lines:
        tags = line.strip().split()
        label = '_'.join(tags[0].split('_')[0:-1])
        x = np.mean([float(tags[1]), float(tags[3])])
        y = np.mean([float(tags[2]), float(tags[4])])
        x_mean = np.mean(pc_x[label])
        x_std = np.std(pc_x[label])
        y_mean = np.mean(pc_y[label])
        y_std = np.std(pc_y[label])
        if not(x_mean - 3 * x_std <= x and x <= x_mean + 3 * x_std and y_mean - 3 * y_std <= y and y <= y_mean + 3 * y_std):
            print(tags[0])
