import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from tqdm import tqdm

#######################

class CSVBenchmarkDataset(Dataset):
    def __init__(self, root, csv_file, transform=None, return_img_paths=False):
        self.root = root
        self.return_img_paths = return_img_paths
        self.data = pd.read_csv(csv_file, delimiter=',')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx].values
        if len(row) == 4:
            if row[-1] == '' or pd.isna(row[-1]):
                img_match = True
                person1, ind1, ind2, _ = row
                person2 = person1
            else:
                img_match = False
                person1, ind1, person2, ind2 = row
        else:
            raise RuntimeError(f"Row can be only of length 4. But found: `{row}`.")
        images = []
        for ind, person in [(ind1, person1),
                            (ind2, person2)]:
            ind = int(ind)
            img_path = self.add_extension(f'{self.root}/{person}/{person}_{ind:04d}')
            if self.return_img_paths:
                img = img_path
            else:
                img = Image.open(img_path)
                if self.transform:
                    img = torch.unsqueeze(torch.tensor(self.transform(img)['pixel_values'][0]), 0)
            images.append(img)
        return *images, img_match

    @staticmethod
    def add_extension(path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError(f'No file `{path}` with extension png or jpg.')

def read_pil_img(img_path, img_sz=112):
    # Чтение изображения и ресайз в размер 112х112
    pil_img = Image.open(img_path).convert('RGB')
    return pil_img.resize((img_sz, img_sz))


def read_img_to_tensor(img_path):
    # Препроцессинг изображения
    pil_img = read_pil_img(img_path)
    img = TF.pil_to_tensor(pil_img)
    img = img.unsqueeze(0).float()
    # img.div_(255).sub_(0.5).div_(0.5)
    img.div_(255)
    return img


@torch.inference_mode()
def norm_single(embed):
    # Нормировка эмбеддинга
    return embed / torch.linalg.norm(embed)


def create_plot(metric,
                out_dir,
                thresholds,
                positive_values, negative_values,
                total_values, cur_epoch):
    # Создание графика зависимости точности на выбранном датасете от выбранного порога.
    thresholds = np.array(thresholds)
    print(f"Saving plot with metric {metric}...")
    plt.title(f'Dependence of accuracy on the selected threshold.\n'
              f'. Metric: {metric}.', wrap=True)
    plt.plot(thresholds, positive_values, color='g', label="positive pairs")
    plt.plot(thresholds, negative_values, color='r', label="negative pairs")
    plt.plot(thresholds, total_values, color='b', label="total pairs")
    plt.legend(loc="lower left")
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')

    major_thresholds_gap = len(thresholds) // 10
    plt.xticks(np.concatenate((thresholds[::major_thresholds_gap], [thresholds[-1]])))
    plt.yticks(np.arange(0, 1.05, 0.1))
    minor_thresholds_gap = len(thresholds) // 20
    plt.xticks(thresholds[::minor_thresholds_gap], minor=True)
    plt.yticks(np.arange(0, 1.05, 0.05), minor=True)
    plt.tick_params(which='minor', length=0)
    plt.grid()
    plt.grid(which='minor', alpha=0.3)

    out_plot_path = f'{out_dir}/accuracy_plot_{metric}_epoch{cur_epoch}.png'
    print(f"out_plot_path: {out_plot_path}")
    plt.savefig(out_plot_path, dpi=500)
    #plt.show()
    plt.close()


def run_benchmarking(dataset, embedder, device, out_plots_dir, cur_epoch):
    embedder.eval()
    start = time.time()

    # Выбор тестируемых порогов для поиска наилучшего для каждой метрики
    metrics = {
        'euclidian': np.arange(0, 3, 0.01),
        'cosine': np.arange(0, 1, 0.01),
    }
    positive_predicted_labels = dict()
    negative_predicted_labels = dict()
    total_predicted_labels = dict()
    for metric in metrics:
        positive_predicted_labels[metric] = {threshold: [] for threshold in metrics[metric]}
        negative_predicted_labels[metric] = {threshold: [] for threshold in metrics[metric]}
        total_predicted_labels[metric] = {threshold: [] for threshold in metrics[metric]}
    true_labels = []
    for image1, image2, issame in tqdm(dataset):
        with torch.no_grad():
            #img1 = read_img_to_tensor(image1).to(device)
            #img2 = read_img_to_tensor(image2).to(device)
            img1 = image1.to(device)
            img2 = image2.to(device)
            # Получение эмбеддингов изображений
            emb1 = embedder(img1).detach()[0]
            emb2 = embedder(img2).detach()[0]
            # Нормировка эмбеддингов изображений
            emb1_norm = norm_single(emb1)
            emb2_norm = norm_single(emb2)
            for metric in metrics:
                if metric == 'cosine':
                    # Подсчет косинусного расстояния
                    dist = 1 - (emb1_norm * emb2_norm).sum()
                else:
                    # Подсчет евклидовой метрики
                    dist = torch.nn.functional.mse_loss(emb1_norm, emb2_norm, reduction='sum')
                dist = dist.item()
                for threshold in metrics[metric]:
                    predicted_label = dist <= threshold
                    total_predicted_labels[metric][threshold].append(predicted_label)
                    if issame:
                        positive_predicted_labels[metric][threshold].append(predicted_label)
                    else:
                        negative_predicted_labels[metric][threshold].append(predicted_label)
            true_labels.append(issame)
        
    cache = dict()
    for metric in metrics:
        print(f"Analyzing thresholds for `{metric}` metric...")
        positive_accuracies = []
        negative_accuracies = []
        total_accuracies = []

        max_ap = (0, None)
        max_acc = (0, None)
        max_f1 = (0, None)
        for threshold in tqdm(metrics[metric]):
            cur_predicted_labels = total_predicted_labels[metric][threshold]
            accuracy = accuracy_score(true_labels, cur_predicted_labels)
            f1_macro = f1_score(true_labels, cur_predicted_labels, average='macro')
            total_accuracies.append(accuracy)
            cur_positive_pred_labels = positive_predicted_labels[metric][threshold]
            positive_accuracies.append(
                accuracy_score([True] * len(cur_positive_pred_labels),
                               cur_positive_pred_labels)
            )
            cur_negative_pred_labels = negative_predicted_labels[metric][threshold]
            negative_accuracies.append(
                accuracy_score([False] * len(cur_negative_pred_labels),
                               cur_negative_pred_labels)
            )
            if accuracy > max_acc[0]:
                max_acc = (accuracy, threshold)
            if f1_macro > max_f1[0]:
                max_f1 = (f1_macro, threshold)
        print("-" * 50)
        print(f"Best Accuracy: {max_acc[0]:.4f} on threshold {max_acc[1]}")
        print(f"Best F1: {max_f1[0]:.4f} on threshold {max_f1[1]}")
        create_plot(metric, out_plots_dir,
                    metrics[metric],
                    positive_accuracies, negative_accuracies, total_accuracies, cur_epoch)
        print("-" * 50)

        cache[metric] = max_f1

    end = time.time() - start
    print(f"Total time: {end} sec")
    print("=" * 50)

    return cache