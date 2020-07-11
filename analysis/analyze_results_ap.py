import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import logging

# params
COST_FP_WEED = 0.3  # Cost incurred, when incorrectly killing 1 corn plant
COST_TP_WEED = 0.5  # Cost saved, when correctly killing 1 weed plant

# setup
FILES = [f for f in os.listdir('ap') if f.endswith('.txt')]  # files with all the data
DATA_REGEX = {'iou_threshold': "mAP@(\d+\.\d+)",
              'conf_threshold': "for conf_thresh = (\d+\.\d+), precision",
              'ap_corn': "name = corn, ap = (\d+\.\d+)",  # average precision, corn
              'ap_weed': "name = weed, ap = (\d+\.\d+)",
              'tp_corn': "name = corn.+\(TP = (\d+)",  # true positives
              'tp_weed': "name = weed.+\(TP = (\d+)",
              'fp_corn': "name = corn.+FP = (\d+)",  # false positives
              'fp_weed': "name = weed.+FP = (\d+)",
              'average_iou': "average IoU.+ (\d+\.\d+)",
              'm_ap': "mean average precision.+ = (\d+\.\d+)",
              'precision': ", precision = (\d+\.\d+),",
              'recall': ", recall = (\d+\.\d+), ",
              'f1': "F1-score = (\d+\.\d+), ",
              'iou_average': "average IoU = (\d+\.\d+), "
              }


def extract_data(FILES: list):
    """
    extract the data from text files
    Parameters
    ----------
    FILES: list of files

    Returns
    -------

    """
    results = []
    for filename in FILES:
        # open data file
        try:
            f = open('results_ap_conf_iou/' + filename, "r")
            text = f.read()
        except Exception as ex:
            logging.error("Error reading file '{}': {}".format(filename, ex))
            raise ex
        # extract data
        for variable_name, variable_regex in DATA_REGEX.items():
            data = {}
            try:
                data[variable_name] = re.compile(variable_regex).findall(text)[0]
            except Exception as ex:
                logging.error("Error getting data from file '{}': {}".format(filename, ex))
                raise ex
            results.append(data)
    df_results = pd.DataFrame(results).astype(float)
    return df_results


def convert_to_numeric(df_results: pd.DataFrame):
    """
    Converts data from results, to numeric. Also performs scaling
    Parameters
    ----------
    df_results: dataframe of results, which were extracted from txt files in extract_data()

    Returns
    -------

    """
    # % to decimal
    df_results[['ap_corn', 'ap_weed', 'average_iou']] = df_results[['ap_corn', 'ap_weed', 'average_iou']] / 100
    # df_results = df_results.sort_values('iou_threshold') # sort rows by iou_threshold
    return df_results


def compute_cost(df_results: pd.DataFrame):
    """
    compute "cost" score
    Parameters
    ----------
    df_results

    Returns
    -------

    """
    df_results["cost_score"] = (COST_TP_WEED * df_results["tp_weed"]) - (COST_FP_WEED * df_results["fp_weed"])
    return df_results


def generate_plots(df_results: pd.DataFrame):
    """
    Generate plots of results
    Parameters
    ----------
    df_results

    Returns
    -------

    """
    # plot - ap
    fig = plt.figure()
    plt.title("Average Precision in Test Set - {}".format(filename))
    plt.plot(df_results['iou_threshold'], df_results['ap_weed'], "-r", label='Weed (AP)')
    plt.plot(df_results['iou_threshold'], df_results['ap_corn'], "-b", label='Corn (AP)')
    plt.plot(df_results['iou_threshold'], df_results['m_ap'], "-g", label='mAP (across both classes)')
    plt.xlabel("IoU Threshold")
    plt.ylabel("Average Precision (AP)")
    plt.legend(loc="lower right")
    fig.savefig('MAP.png')

    # plot - fp weeds
    fig = plt.figure()
    plt.title("False Positives (Weeds) in Test Set - {}".format(filename))
    plt.plot(df_results['iou_threshold'], df_results['fp_weed'], "-b", label='Weed (FP)')
    plt.plot(df_results['iou_threshold'], df_results['fp_corn'], "-r", label='Corn (FP)')
    plt.xlabel("IoU Threshold")
    plt.ylabel("# of False Positives (FP)")
    # plot - tp weeds (on same fig)
    plt.plot(df_results['iou_threshold'], df_results['tp_weed'], "-o", color="b", label='Weed (TP)')
    plt.plot(df_results['iou_threshold'], df_results['tp_corn'], "-o", color="r", label='Corn (TP)')
    plt.legend(loc="lower right")
    fig.savefig('TPFP.png')

    # PR curve
    fig = plt.figure()
    plt.title("Precision-Recall Curve across different IoU thresholds - \n for filename {}".format(filename))
    plt.plot(df_results['recall'], df_results['precision'])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    fig.savefig('Precision-Recall.png')

    # Recall-IoU curve
    fig = plt.figure()
    plt.title("Recall-IoU Curve - \n for filename {}".format(filename))
    plt.plot(df_results['iou_threshold'], df_results['recall'], "-b", label="Recall")
    plt.plot(df_results['iou_threshold'], df_results['precision'], "-r", label="Precision")
    plt.xlabel("IoU threshold")
    plt.legend(loc="lower right")
    fig.savefig('Recall-Precision-IoU.png')

    # Cost curve
    fig = plt.figure()
    plt.title("Cost - \n for filename {}".format(filename))
    plt.plot(df_results['iou_threshold'], df_results['cost_score'], "-b", label="cost")
    plt.xlabel("IoU threshold")
    plt.legend(loc="lower right")
    fig.savefig('Recall-Precision-IoU.png')
