import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

files = [f for f in os.listdir('fps') if f.endswith('.txt')]

for filename in files:

    f = open('fps/' + filename, "r")
    text = f.read()

    # get FPS
    df_fps = pd.DataFrame(re.compile('AVG_FPS:(\d+\.\d+)').findall(text)).rename(columns={0: 'avg_fps'})  # avg fps
    df_fps['fps'] = re.compile(r'\nFPS:(\d+\.\d+)').findall(text)  # fps
    df_fps = df_fps.astype(float)  # convert to float
    df_fps = df_fps[df_fps['fps'] != 0]  # remove invalid entries
    df_fps.rename(columns={'index': 'idx'}, inplace=True)

    # get FPS, cumulative mean
    df_fps['cummean_raw'] = df_fps['fps'].expanding().mean()  # raw cumulative mean
    warmup_cutoff = df_fps['cummean_raw'].mean() - 2*df_fps['cummean_raw'].std()  # remove warm-up values, which lowers FPS
    df_fps['cummean_filtered'] = np.nan
    df_fps['cummean_filtered'][df_fps['fps'] > warmup_cutoff] = \
        df_fps['fps'][df_fps['fps'] > warmup_cutoff].expanding().mean()

    # plot FPS, analysis
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

    # get actual number of frames & duration of video
    """https://stackoverflow.com/questions/49048111/how-to-get-the-duration-of-video-using-cv2"""
    cap = cv2.VideoCapture("2019_07_24_1_Up_Crash.MP4")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("num frames: {}".format(num_frames))
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    print("fps: {}".format(fps))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration / 60)
    seconds = duration % 60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
    cap.release()


    # get confidence