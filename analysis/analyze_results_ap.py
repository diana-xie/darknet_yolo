import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

files = [f for f in os.listdir('ap') if f.endswith('.txt')]
results = []

for filename in files:

    try:
        f = open('ap/' + filename, "r")
        text = f.read()
    except Exception as ex:
        logging.error("Error reading file '{}': {}".format(filename, ex))
        raise ex

    # get data
    try:
        data = {'iou_threshold': re.compile(r"mAP@(\d+\.\d+)").findall(text)[0],
                'ap_corn': re.compile(r"name = corn, ap = (\d+\.\d+)").findall(text)[0],  # average precision, corn
                'ap_weed': re.compile(r"name = weed, ap = (\d+\.\d+)").findall(text)[0],
                'tp_corn': re.compile(r"name = corn.+\(TP = (\d+)").findall(text)[0],  # true positives
                'tp_weed': re.compile(r"name = weed.+\(TP = (\d+)").findall(text)[0],
                'fp_corn': re.compile(r"name = corn.+FP = (\d+)").findall(text)[0],  # false positives
                'fp_weed': re.compile(r"name = weed.+FP = (\d+)").findall(text)[0],
                'average_iou': re.compile(r"average IoU.+ (\d+\.\d+)").findall(text)[0],
                'm_ap': re.compile(r"mean average precision.+ = (\d+\.\d+)").findall(text)[0]}
        results.append(data)
    except Exception as ex:
        logging.error("Error getting data from file '{}': {}".format(filename, ex))
        raise ex

# convert to numeric
df_ap = pd.DataFrame(results).astype(float)
df_ap[['ap_corn', 'ap_weed', 'average_iou']] = df_ap[['ap_corn', 'ap_weed', 'average_iou']]/100  # % to decimal
df_ap = df_ap.sort_values('iou_threshold')

# plot - ap
fig = plt.figure()
plt.title("Average Precision in Test Set".format(filename))
plt.plot(df_ap['iou_threshold'], df_ap['ap_weed'], "-r", label='Weed (AP)')
plt.plot(df_ap['iou_threshold'], df_ap['ap_corn'], "-b", label='Corn (AP)')
plt.xlabel("IoU Threshold")
plt.ylabel("Average Precision (AP)")
plt.legend(loc="lower right")
fig.savefig('AP_analysis.png')

# plot - fp weeds
fig = plt.figure()
plt.title("False Positives (Weeds) in Test Set".format(filename))
plt.plot(df_ap['iou_threshold'], df_ap['fp_weed'], "-g", label='Weed (FP)')
plt.plot(df_ap['iou_threshold'], df_ap['fp_corn'], "-y", label='Corn (FP)')
plt.xlabel("IoU Threshold")
plt.ylabel("# of False Positives (FP)")
plt.legend(loc="lower right")
fig.savefig('FP_analysis.png')
