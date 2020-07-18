import os
import re
import pandas as pd
import numpy as np
import logging

# internal imports
from analysis.generate_plots import make_save_plots

# setup
DATA_REGEX = {'iou_threshold': "mAP@(\d+\.\d+)",
              'conf_threshold': "for conf_thresh = (\d+\.\d+), precision",
              'ap_corn': "name = corn, ap = (\d+\.\d+)",  # average precision, corn
              'ap_weed': "name = weed, ap = (\d+\.\d+)",
              'tp_corn': "name = corn.+\(TP = (\d+)",  # true positives
              'tp_weed': "name = weed.+\(TP = (\d+)",
              'fp_corn': "name = corn.+FP = (\d+)",  # false positives
              'fp_weed': "name = weed.+FP = (\d+)",
              'average_iou': "average IoU = (\d+\.\d+)",
              'm_ap': "mean average precision.+ = (\d+\.\d+)",
              'precision': ", precision = (\d+\.\d+),",
              'recall': ", recall = (\d+\.\d+), ",
              'f1': "F1-score = (\d+\.\d+)"
              }


def extract_data(FILES: list):
    """
    extract the data from text files
    Parameters
    ----------
    FILES: list of files

    """
    # extract data
    results = []
    for filename in FILES:
        # open data file
        try:
            f = open('results_ap_conf_iou/' + filename, "r")
            text = f.read()
        except Exception as ex:
            logging.error("Error reading file '{}': {}".format(filename, ex))
            raise ex
        # extract data using regex
        data_extracted = {}
        for variable_name, variable_regex in DATA_REGEX.items():
            try:
                data_extracted[variable_name] = re.compile(variable_regex).findall(text)[0]
            except Exception as ex:
                logging.error("Error getting data from file '{}': {}".format(filename, ex))
                data_extracted[variable_name] = 'nan'
        results.append(data_extracted)

    # postprocessing
    results = pd.DataFrame(results).astype(float)
    results = results[results['precision'].notnull()]

    return results


def convert_to_numeric(data: pd.DataFrame):
    """
    Converts data from results, to numeric. Also performs scaling
    Parameters
    ----------
    data: dataframe of results, which were extracted from txt files in extract_data()

    """
    data[['ap_corn', 'ap_weed', 'average_iou']] = data[['ap_corn', 'ap_weed', 'average_iou']] / 100  # % to decimal
    return data


def compute_cost(data: pd.DataFrame, gain_tp_weed: float = 0.5, cost_fp_weed: float = 0.3):
    """
    compute "cost" score
    Parameters
    ----------
    data: df of results, extracted from txt files of inference
    gain_tp_weed: Cost saved, when correctly killing 1 weed plant
    cost_fp_weed: Cost incurred, when incorrectly killing 1 corn plant

    Returns
    -------
    df with new "cost_score" col.positive values indicate $ gained. negative means it's losing $
    """
    data['cost_score'] = (gain_tp_weed * data["tp_weed"]) - (cost_fp_weed * data["fp_weed"])
    data['gain_tp_weed'] = gain_tp_weed
    data['cost_fp_weed'] = cost_fp_weed
    return data


def grid_cost(data: pd.DataFrame):
    costs = []
    for gain_tp_weed in np.arange(0.1, 1.1, 0.1):
        for cost_fp_weed in np.arange(0.1, 1.1, 0.1):
            df = compute_cost(data=data, gain_tp_weed=gain_tp_weed, cost_fp_weed=cost_fp_weed)
            costs.append(df)
    costs = pd.concat(costs)
    costs = costs.groupby(['iou_threshold', 'conf_threshold'])['cost_score'].max().reset_index()
    return costs


if __name__ == "__main__":
    FILES = [f for f in os.listdir('results_ap_conf_iou') if f.endswith('.txt')]  # files with all the data
    df_extracted = extract_data(FILES)
    df_results = convert_to_numeric(data=df_extracted)
    df_costs = grid_cost(data=df_results)
    make_save_plots(df_results=df_results, df_costs=df_costs)
