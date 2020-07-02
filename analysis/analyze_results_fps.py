import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# iou_threshold = 24
# filename = "result_iouthresh_{}.txt".format(str(iou_threshold))
files = [f for f in os.listdir() if f.endswith('.txt')]

for filename in files:

    f = open(filename, "r")
    text = f.read()

    # make FPS table
    df_fps = pd.DataFrame(re.compile('AVG_FPS:(\d+\.\d+)').findall(text))  # get avg_fps col
    df_fps['fps'] = re.compile(r'\nFPS:(\d+\.\d+)').findall(text)  # get fps col

    # pre-processing
    df_fps = df_fps.astype(float)
    df_fps = df_fps[df_fps['fps'] != 0]
    df_fps = df_fps.rename(columns={0: 'avg_fps'}).reset_index()
    df_fps.rename(columns={'index': 'idx'}, inplace=True)

    # get cumulative mean
    df_fps['cummean_raw'] = df_fps['fps'].expanding().mean()  # raw cumulative mean
    warmup_cutoff = df_fps['cummean_raw'].mean() - 2*df_fps['cummean_raw'].std()  # remove warm-up values, which lowers FPS
    df_fps['cummean_filtered'] = np.nan
    df_fps['cummean_filtered'][df_fps['fps'] > warmup_cutoff] = \
        df_fps['fps'][df_fps['fps'] > warmup_cutoff].expanding().mean()

    # get total number of frames


    # plot
    fig = plt.figure()
    plt.title("FPS in '{}'".format(filename))
    plt.plot(df_fps['idx'], df_fps['avg_fps'], "-b", label='FPS (mean)')
    plt.plot(df_fps['idx'], df_fps['fps'], "-r", label='FPS (raw)')
    plt.plot(df_fps['idx'], df_fps['cummean_raw'], "-o", label='FPS, cumulative mean (raw)')
    plt.plot(df_fps['idx'], df_fps['cummean_filtered'], "-p", label='FPS, cumulative mean (after warmup phase)')
    plt.xlabel("frame #")
    plt.ylabel("FPS or average FPS")
    plt.legend(loc="lower right")
    fig.savefig('FPS_analysis_{}.png'.format(filename))
