""" Generate plots of results """

import matplotlib.pyplot as plt
import pandas as pd


def ap_plot(df_results: pd.DataFrame):
    """
    Generates IoU threshold vs. AP/mAP plot
    Parameters
    """
    try:
        # plot - ap
        fig = plt.figure()
        plt.title("Average Precision in Test Set - {}")
        # plt.plot(df_results['iou_threshold'], df_results['ap_weed'], "-r", label='Weed (AP)')
        # plt.plot(df_results['iou_threshold'], df_results['ap_corn'], "-b", label='Corn (AP)')
        # plt.plot(df_results['iou_threshold'], df_results['m_ap'], "-g", label='mAP (across both classes)')
        plt.scatter(df_results['iou_threshold'], df_results['ap_weed'], c="r", label='Weed (AP)')
        plt.scatter(df_results['iou_threshold'], df_results['ap_corn'], c="b", label='Corn (AP)')
        plt.scatter(df_results['iou_threshold'], df_results['m_ap'], c="g", label='mAP (across both classes)')
        plt.xlabel("IoU Threshold")
        plt.ylabel("Average Precision (AP)")
        plt.legend(loc="upper right")
        fig.savefig('all_MAP.png')  # all means generating pred for range of IoU thresholds + confidence thresholds
    except Exception as ex:
        print("Error in generating 'IoU threshold vs. AP/mAP' plot: {}".format(ex))


def tp_fp_plot(df_results: pd.DataFrame):

    """
    Generate plot of IoU threshold vs. TP weed/corn
    Parameters
    """
    try:
        # plot - fp weeds
        fig = plt.figure()
        plt.title("False Positives (Weeds) in Test Set")
        plt.plot(df_results['iou_threshold'], df_results['fp_weed'], "-b", label='Weed (FP)')
        plt.plot(df_results['iou_threshold'], df_results['fp_corn'], "-r", label='Corn (FP)')
        plt.xlabel("IoU Threshold")
        plt.ylabel("# of False Positives (FP)")
        # plot - tp weeds (on same fig)
        plt.plot(df_results['iou_threshold'], df_results['tp_weed'], "-o", color="b", label='Weed (TP)')
        plt.plot(df_results['iou_threshold'], df_results['tp_corn'], "-o", color="r", label='Corn (TP)')
        plt.legend(loc="lower right")
        fig.savefig('all_TPFP.png')
    except Exception as ex:
        print("Error in generating 'IoU threshold vs. TP weed/corn' plot: {}".format(ex))


def pr_plot(df_results: pd.DataFrame):
    """
    Generates Precision-Recall plot
    Parameters
    """
    try:
        # PR curve
        fig = plt.figure()
        plt.title("Precision-Recall Curve across different IoU thresholds")
        plt.plot(df_results['recall'], df_results['precision'])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        fig.savefig('Precision-Recall.png')
    except Exception as ex:
        print("Error in generating Precision-Recall plot: {}".format(ex))


def recall_iou_plot(df_results: pd.DataFrame):
    """
    Generate IoU threshold vs. Recall/Precision plot
    """
    try:
        # Recall-IoU curve
        fig = plt.figure()
        plt.title("Recall & Precision vs. IoU Curve")
        plt.plot(df_results['iou_threshold'], df_results['recall'], "-b", label="Recall")
        plt.plot(df_results['iou_threshold'], df_results['precision'], "-r", label="Precision")
        plt.xlabel("IoU threshold")
        plt.legend(loc="lower right")
        fig.savefig('Recall-Precision-IoU.png')
    except Exception as ex:
        print("Error in generating 'IoU threshold vs. Recall/Precision' plot: {}".format(ex))


def cost_plot(df_costs: pd.DataFrame):
    """
    Generate  plot
    """
    try:
        # Cost curve
        fig = plt.figure()
        plt.title("IoU threshold vs. Cost")
        # plt.plot(df_results['iou_threshold'], df_results['cost_score'], "-b", label="cost")
        plt.scatter(df_costs['iou_threshold'], df_costs['cost_score'], c="b", label="IoU")
        plt.scatter(df_costs['conf_threshold'], df_costs['cost_score'], c="r", label="Confidence")
        plt.xlabel("IoU/Confidence threshold")
        plt.ylabel("Cost (USD)")
        plt.legend(loc="lower center")
        fig.savefig('IoU_cost.png')
    except Exception as ex:
        print('Error in generating cost plot: {}'.format(ex))


def make_save_plots(df_results: pd.DataFrame, df_costs: pd.DataFrame):
    """
    Generates & saves plots of results
    Parameters
    ----------
    df_results: df of results
    df_costs: df of costs, from df_results

    Returns
    -------
    Generates & saves plots of results
    """
    ap_plot(df_results=df_results)
    tp_fp_plot(df_results=df_results)
    pr_plot(df_results=df_results)
    recall_iou_plot(df_results=df_results)
    cost_plot(df_costs=df_costs)

