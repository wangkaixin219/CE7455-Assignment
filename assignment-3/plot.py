import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time


def show_plot(points):
    print('showPlot')
    plt.plot(points)
    plt.ylabel('Loss', fontsize=15)
    fig = plt.gcf()
    fig.savefig('Loss.png', dpi=1000)
    plt.show()


def as_minutes(s):
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    # return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    return '%s' % (as_minutes(s))
