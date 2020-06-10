import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

f = open("result_iouthresh_024.txt", "r")
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

# plot
fig = plt.figure()
plt.title("FPS in '2019_07_24_1_Up_Crash.mp4'")
plt.plot(df_fps['idx'], df_fps['avg_fps'], "-b", label='FPS (mean)')
plt.plot(df_fps['idx'], df_fps['fps'], "-r", label='FPS (raw)')
plt.plot(df_fps['idx'], df_fps['cummean_raw'], "-o", label='FPS, cumulative mean (raw)')
plt.plot(df_fps['idx'], df_fps['cummean_filtered'], "-p", label='FPS, cumulative mean (after warmup phase)')
plt.xlabel("frame #")
plt.ylabel("FPS or average FPS")
plt.legend(loc="lower right")
fig.savefig('FPS_analysis.png')