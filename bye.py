import json
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import argparse
import os

def load_entropy_matrix(jsonl_path, num_layers=32):
    entropy_matrix = []
    question_ids = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            entropy_dict = data["entropy"]
            entropy_vec = [entropy_dict[str(l)] for l in range(num_layers)]
            entropy_matrix.append(entropy_vec)
            question_ids.append(data["question_id"])

    entropy_matrix = np.array(entropy_matrix)
    if np.isnan(entropy_matrix).any():
        print(f"NaN values detected in entropy_matrix, replacing with column mean.")
        entropy_matrix = np.nan_to_num(entropy_matrix, nan=np.nanmean(entropy_matrix, axis=0))

    return entropy_matrix, question_ids

def compute_kl_between_gaussians(mean1, std1, mean2, std2):
    var1 = std1 ** 2
    var2 = std2 ** 2
    return np.log(std2 / std1) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5

def check_bimodal_distribution(gmm):
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()

    mean_diff = abs(means[0] - means[1])
    if mean_diff > 2.0 * (stds[0] + stds[1]):
        return True
    return False

def get_layerwise_kl(entropy_matrix):
    num_layers = entropy_matrix.shape[1]
    kl_list = []
    sensitive_layers = []

    for l in range(num_layers):
        values = entropy_matrix[:, l].reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(values)
        
        if check_bimodal_distribution(gmm):
            sensitive_layers.append(l)
        
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_).flatten()

        if means[0] > means[1]:
            means = means[::-1]
            stds = stds[::-1]

        kl = compute_kl_between_gaussians(means[0], stds[0], means[1], stds[1])
        kl_list.append(kl)

    return np.array(kl_list), sensitive_layers

def detect_suspicious_samples(entropy_matrix, question_ids, sensitive_layers):
    suspicious_scores = entropy_matrix[:, sensitive_layers].mean(axis=1)
    
    gmm = GaussianMixture(n_components=2, random_state=0)
    cluster_labels = gmm.fit_predict(suspicious_scores.reshape(-1, 1))
    cluster_means = gmm.means_.flatten()
    low_entropy_cluster = np.argmin(cluster_means)
    suspicious_indices = np.where(cluster_labels == low_entropy_cluster)[0].tolist()
    suspicious_question_ids = [question_ids[i] for i in suspicious_indices]
    
    return suspicious_indices, suspicious_question_ids, suspicious_scores, cluster_labels

def evaluate_detection(suspicious_ids, all_ids):
    true_positives = [qid for qid in suspicious_ids if "poison" in str(qid)]
    false_positives = [qid for qid in suspicious_ids if "poison" not in str(qid)]
    total_actual_backdoor = len([qid for qid in all_ids if "poison" in str(qid)])

    precision = len(true_positives) / (len(true_positives) + len(false_positives) + 1e-8)
    recall = len(true_positives) / (total_actual_backdoor + 1e-8)

    return precision, recall

def detect_backdoor_samples(jsonl_path):
    entropy_matrix, question_ids = load_entropy_matrix(jsonl_path)
    kl_list, sensitive_layers = get_layerwise_kl(entropy_matrix)
    
    if not sensitive_layers:
        print("No sensitive layers found based on bimodal distribution.")
        return None
    
    suspicious_indices, suspicious_question_ids, suspicious_scores, cluster_labels = detect_suspicious_samples(
        entropy_matrix, question_ids, sensitive_layers)

    return {
        "kl_per_layer": kl_list.tolist(),
        "sensitive_layers": sensitive_layers,
        "suspicious_sample_indices": suspicious_indices,
        "suspicious_question_ids": suspicious_question_ids,
        "all_question_ids": question_ids,
        "entropy_matrix": entropy_matrix,
        "suspicious_scores": suspicious_scores,
        "cluster_labels": cluster_labels
    }

def remove_suspicious_samples_from_json(original_json_path, output_json_path, suspicious_ids):
    with open(original_json_path, 'r') as f:
        data = json.load(f)

    cleaned_data = [item for item in data if str(item["id"]) not in suspicious_ids]

    with open(output_json_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"Removed {len(data) - len(cleaned_data)} suspicious samples.")
    print(f"Cleaned file saved to: {output_json_path}")


def plot_kl_divergence(args, kl_list):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(kl_list)), kl_list, marker='o')
    plt.title("KL Divergence per Layer")
    plt.xlabel("Layer Index")
    plt.ylabel("KL Divergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.entropy_file, "kl_divergence.png"))

def plot_suspicious_score_distribution(args, scores):
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=100, color='skyblue', edgecolor='black')
    plt.title("Distribution of Suspicious Scores (Lower = More Suspicious)")
    plt.xlabel("Suspicious Score (Avg. Attention Entropy)")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.entropy_file, "suspicious_score_distribution.png"))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Detect backdoor samples in a JSONL file.")
    argparser.add_argument("--entropy-file", type=str)
    argparser.add_argument("--train-file", type=str)
    argparser.add_argument("--clean-file", type=str)
    args = argparser.parse_args()

    jsonl_path = os.path.join(args.entropy_file, "1-entropy.jsonl")
    result = detect_backdoor_samples(jsonl_path)

    remove_suspicious_samples_from_json(args.train_file, args.clean_file, result["suspicious_question_ids"])

    precision, recall = evaluate_detection(result["suspicious_question_ids"], result["all_question_ids"])

    with open(os.path.join(args.entropy_file, "filter-result.txt"), "w") as f:
        f.write(f"Sensitive Layers: {result['sensitive_layers']}\n")
        f.write(f"Suspicious Sample Indices: {result['suspicious_sample_indices']}\n")
        f.write(f"Suspicious Question IDs: {result['suspicious_question_ids']}\n")
        f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}\n")

    plot_kl_divergence(args, result["kl_per_layer"])
    plot_suspicious_score_distribution(args, result["suspicious_scores"])