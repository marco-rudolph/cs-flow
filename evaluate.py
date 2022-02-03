import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve
from tqdm import tqdm
from model import load_model, FeatureExtractor
import config as c
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import PIL
from os.path import join
import os
from copy import deepcopy
from math import isclose


localize = True
upscale_mode = 'bilinear'
score_export_dir = join('./viz/scores/', c.modelname)
os.makedirs(score_export_dir, exist_ok=True)
map_export_dir = join('./viz/maps/', c.modelname)
os.makedirs(map_export_dir, exist_ok=True)


def compare_histogram(scores, classes, thresh=2.5, n_bins=64):
    classes = deepcopy(classes)
    classes[classes > 0] = 1
    scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(0.5, thresh, 5)
    labels = [str(i) for i in ticks[:-1]] + ['>' + str(thresh)]
    plt.xticks(ticks, labels=labels)
    plt.xlabel(r'$-log(p(z))$')
    plt.ylabel('Count (normalized)')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(join(score_export_dir, 'score_histogram.png'), bbox_inches='tight', pad_inches=0)


def viz_roc(values, classes, class_names):
    def export_roc(values, classes, export_name='all'):
        # Compute ROC curve and ROC area for each class
        classes = deepcopy(classes)
        classes[classes > 0] = 1
        fpr, tpr, _ = roc_curve(classes, values)
        roc_auc = auc(fpr, tpr)

        plt.clf()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class ' + c.class_name)
        plt.legend(loc="lower right")
        plt.axis('equal')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.savefig(join(score_export_dir, export_name + '.png'))

    export_roc(values, classes)
    for cl in range(1, classes.max() + 1):
        filtered_indices = np.concatenate([np.where(classes == 0)[0], np.where(classes == cl)[0]])
        classes_filtered = classes[filtered_indices]
        values_filtered = values[filtered_indices]
        export_roc(values_filtered, classes_filtered, export_name=class_names[filtered_indices[-1]])


def viz_maps(maps, name, label):
    img_path = img_paths[c.viz_sample_count]
    image = PIL.Image.open(img_path).convert('RGB')
    image = np.array(image)

    map_to_viz = t2np(F.interpolate(maps[0][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
        0, 0]

    plt.clf()
    plt.imshow(map_to_viz)
    plt.axis('off')
    plt.savefig(join(map_export_dir, name + '_map.jpg'), bbox_inches='tight', pad_inches=0)

    if label > 0:
        plt.clf()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(join(map_export_dir, name + '_orig.jpg'), bbox_inches='tight', pad_inches=0)
        plt.imshow(map_to_viz, cmap='viridis', alpha=0.3)
        plt.savefig(join(map_export_dir, name + '_overlay.jpg'), bbox_inches='tight', pad_inches=0)
    return


def viz_map_array(maps, labels, n_col=8, subsample=4, max_figures=-1):
    plt.clf()
    fig, subplots = plt.subplots(3, n_col)

    fig_count = -1
    col_count = -1
    for i in range(len(maps)):
        if i % subsample != 0:
            continue

        if labels[i] == 0:
            continue

        col_count = (col_count + 1) % n_col
        if col_count == 0:
            if fig_count >= 0:
                plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
                plt.close()
            fig, subplots = plt.subplots(3, n_col, figsize=(22, 8))
            fig_count += 1
            if fig_count == max_figures:
                return

        anomaly_description = img_paths[i].split('/')[-2]
        image = PIL.Image.open(img_paths[i]).convert('RGB')
        image = np.array(image)
        map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
            0, 0]
        subplots[1][col_count].imshow(map)
        subplots[1][col_count].axis('off')
        subplots[0][col_count].imshow(image)
        subplots[0][col_count].axis('off')
        subplots[0][col_count].set_title(c.class_name + ":\n" + anomaly_description)
        subplots[2][col_count].imshow(image)
        subplots[2][col_count].axis('off')
        subplots[2][col_count].imshow(map, cmap='viridis', alpha=0.3)
    for i in range(col_count, n_col):
        subplots[0][i].axis('off')
        subplots[1][i].axis('off')
        subplots[2][i].axis('off')
    if col_count > 0:
        plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
    return


def evaluate(model, test_loader):
    model.to('cuda')
    model.eval()
    if not c.pre_extracted:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False

    print('\nCompute maps, loss and scores on test set:')
    anomaly_score = list()
    test_labels = list()
    c.viz_sample_count = 0
    all_maps = list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            if not c.pre_extracted:
                inputs = fe(inputs)
            z = model(inputs)

            z_concat = t2np(concat_maps(z))
            nll_score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
            anomaly_score.append(nll_score)
            test_labels.append(t2np(labels))

            if localize:
                z_grouped = list()
                likelihood_grouped = list()
                for i in range(len(z)):
                    z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                    likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)) / c.n_feat)
                all_maps.extend(likelihood_grouped[0])
                for i_l, l in enumerate(t2np(labels)):
                    # viz_maps([lg[i_l] for lg in likelihood_grouped], c.modelname + '_' + str(c.viz_sample_count), label=l, show_scales = 1)
                    c.viz_sample_count += 1

    anomaly_score = np.concatenate(anomaly_score)
    test_labels = np.concatenate(test_labels)
    compare_histogram(anomaly_score, test_labels)

    class_names = [img_path.split('/')[-2] for img_path in img_paths]
    viz_roc(anomaly_score, test_labels, class_names)
    test_labels = np.array([1 if l > 0 else 0 for l in test_labels])

    # General metric
    auc_score = roc_auc_score(test_labels, anomaly_score)
    AP = average_precision_score(test_labels, anomaly_score)

    # Optimal threshold
    precision, recall, thresholds = precision_recall_curve(test_labels, anomaly_score)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    idx = f1.argmax()
    opt_precision, opt_recall, opt_f1, opt_thresh = precision[idx], recall[idx], f1[idx], thresholds[idx]

    # Precision at full recall
    _recall = recall.copy()
    _recall.sort()
    max_recall_idx = _recall.argmax()
    full_recall = recall[max_recall_idx]
    precision_fullRec = precision[max_recall_idx]

    print('AUROC:', auc_score)
    print('AP:', AP)
    print()
    print('Optimal F1:', opt_f1)
    print('Precision:', opt_precision)
    print('Recall:', opt_recall)
    print('Threshold:', opt_thresh)
    print()
    print('Full recall:', full_recall)
    print('Highest precision at full recall:', precision_fullRec)

    if localize:
        viz_map_array(all_maps, test_labels, n_col=4, subsample=1)

    # Plot AUC
    fpr, tpr, thresholds = roc_curve(test_labels, anomaly_score)
    plt.figure('auroc_curve')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend([f'AUC = {auc_score:.4f}'])
    for idx in range(len(thresholds)):
        if idx > 0:
            if isclose(tpr[idx], tpr[idx - 1], abs_tol=1e-2):
                continue
        plt.plot([fpr[idx]], [tpr[idx]], 'o', color='green')
        plt.text(fpr[idx] + 0.01, tpr[idx] - 0.04,
                 'fpr: ' + str(round(fpr[idx], 2)) + '_tpr: ' + str(round(tpr[idx], 2)) + '_th: ' + str(round(thresholds[idx], 2)), fontsize='x-small')

    # Precision at full recall
    full_recall_idx = tpr.argmax()
    plt.plot([fpr[full_recall_idx]], [tpr[full_recall_idx]], 'o', color='green')
    plt.text(fpr[full_recall_idx] + 0.01, tpr[full_recall_idx] - 0.04,
             'fpr: ' + str(round(fpr[full_recall_idx], 2)) + '_tpr: ' + str(round(tpr[full_recall_idx], 2)) + '_th: ' + str(
                 round(thresholds[full_recall_idx], 2)), fontsize='x-small')

    plt.title(c.modelname)
    plt.savefig(join(score_export_dir, 'auroc_curve.png'))

    return


train_set, test_set = load_datasets(c.dataset_path, c.class_name)
img_paths = test_set.paths if c.pre_extracted else [p for p, l in test_set.samples]
_, test_loader = make_dataloaders(train_set, test_set)
mod = load_model(c.modelname)
evaluate(mod, test_loader)