import matplotlib.pyplot as plt
import os
import numpy as np


def getXYFromFile(path):
    with open(path) as f:
        data = f.readlines()
        y = []
        x = []
        for line in data:
            if 'best' in line:
                data = line.split(' ')
                epoch = data[3].split(':')[-1]
                err_mad = data[7].split(':')[-1]
                err_mad = float(err_mad)
                epoch = float(epoch)
                y.append(err_mad)
                x.append(epoch)
        end = int(x[-1])
        # if end != 499:
        #     end += int(x[-end-2]) + 499
        x = x[-end - 1:-end - 1]
        y = y[-end - 1:-end - 1]
    return np.array(x), np.array(y)


if __name__ == '__main__':
    th = 0.05
    sc1 = 5
    sc2 = 2
    max_y = 0.2
    x, y = getXYFromFile('/data/wjw/work/matting_tool_study/checkSave/bfd/PPM/17/log.txt')
    y[y > th] = sc1 + sc2 * (y[y > th] - min(y[y > th])) / (max_y - min(y[y > th]))
    y[y <= th] = sc1 * (y[y < th]) / (max(y[y < th]))
    plt.plot(x, y, label='17')
    # x, y = getXYFromFile('/data/wjw/work/matting_tool_study/checkSave/bfd/PPM/19/log.txt')
    # y[y > th] = sc1 + sc2 * (y[y > th] - min(y[y > th])) / (max_y - min(y[y > th]))
    # y[y <= th] = sc1 * (y[y < th]) / (max(y[y < th]))
    # plt.plot(x, y, label='19')
    x, y = getXYFromFile('/data/wjw/work/matting_tool_study/checkSave/bfd/PPM/18/log.txt')
    y[y > th] = sc1 + sc2 * (y[y > th] - min(y[y > th])) / (max_y - min(y[y > th]))
    y[y <= th] = sc1 * (y[y < th]) / (max(y[y < th]))
    plt.plot(x, y, label='18')
    plt.legend(loc=0)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['0', '0.02', '0.04', '0.06', '0.08', '0.1', str(max_y / 2.0), str(max_y)])
    plt.grid()
    plt.show()
