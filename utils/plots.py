

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


def tm_class_sums_to_prob(class_sums, T):
    """
    Converts Tsetlin Machine class sums to probabilities.
    """
    return 0.5 * (1.0 + class_sums / T)

def plot_tm_roc_curve_multi_specialist(
    config,
    params,
    filename_prefixes,
    specialist_names,
    Y_test,
    class_names=None,
    figsize=(6, 5),
    save_path=None,
    label_fontsize=10,
    legend_fontsize=9,
):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    # load and convert scores for each specialist
    all_probs = []
    for p, prefix in zip(params, filename_prefixes):
        file_path = (
            f"{config.SAVE_DIR_SCORES}/"
            f"{prefix}_{config.N_EPOCHS}_"
            f"{config.NUM_CLAUSES}_"
            f"{p['T_best']}_"
            f"{p['s_best']:.1f}_"
            f"{p['patch_best']}_"
            f"{p['maxlit_best']}_"
            f"{p['weighted_best']}.txt"
        )
        Y_test_class_sums = np.loadtxt(file_path, delimiter=",")
        Y_test_probs = tm_class_sums_to_prob(Y_test_class_sums, p['T_best'])
        print(f"\n[Loading] {prefix}")
        print(f"  Class sums shape: {Y_test_class_sums.shape}")
        print(f"  Probs shape:      {Y_test_probs.shape}")
        print(f"  Probs sample (first 3 rows):\n{Y_test_probs[:3]}")
        all_probs.append(Y_test_probs)

    n_classes = all_probs[0].shape[1]
    if class_names is None:
        class_names = [f"Class {c}" for c in range(n_classes)]

    # binarize ground truth labels
    if n_classes == 2:
        Y_true = (Y_test == 1).astype(int)
        Y_test_binarized = np.column_stack([1 - Y_true, Y_true])
    else:
        Y_test_binarized = label_binarize(Y_test, classes=list(range(n_classes)))

    print(f"\n[Ground truth]")
    print(f"  Y_test unique values:      {np.unique(Y_test)}")
    print(f"  Y_test_binarized shape:    {Y_test_binarized.shape}")
    print(f"  Y_test_binarized (first 3 rows):\n{Y_test_binarized[:3]}")

    # report macro AUC and per-class AUC per specialist
    print("\n" + "=" * 50)
    for name, probs in zip(specialist_names, all_probs):
        print(f"\n=== {name} ===")
        if n_classes == 2:
            print(f"  [Binary] computing AUC per class:")
            print(f"    class 0 ({class_names[0]}): y_true shape {Y_test_binarized[:, 0].shape}, y_score shape {probs[:, 0].shape}")
            auc_0 = auc(*roc_curve(Y_test_binarized[:, 0], probs[:, 0])[:2])
            print(f"    class 1 ({class_names[1]}): y_true shape {Y_test_binarized[:, 1].shape}, y_score shape {probs[:, 1].shape}")
            auc_1 = auc(*roc_curve(Y_test_binarized[:, 1], probs[:, 1])[:2])
            specialist_macro = (auc_0 + auc_1) / 2
            print(f"  Macro AUC: {specialist_macro:.4f}  (mean of {class_names[0]}: {auc_0:.3f} and {class_names[1]}: {auc_1:.3f})")
        else:
            print(f"  [Multiclass] y_true shape: {Y_test_binarized.shape}, y_score shape: {probs.shape}")
            specialist_macro = roc_auc_score(
                Y_test_binarized, probs,
                average="macro", multi_class="ovr"
            )
            print(f"  Macro AUC: {specialist_macro:.4f}")
            for c in range(n_classes):
                print(f"    class {c} ({class_names[c]}): y_true shape {Y_test_binarized[:, c].shape}, y_score shape {probs[:, c].shape}")
                fpr, tpr, _ = roc_curve(Y_test_binarized[:, c], probs[:, c])
                class_auc = auc(fpr, tpr)
                print(f"    {class_names[c]}: AUC = {class_auc:.3f}")
    print("=" * 50)

    colors = [
        plt.cm.Blues(0.7),
        "#FFD700",
        plt.cm.Oranges(0.7),
        plt.cm.Purples(0.7),
        plt.cm.Greens(0.7),
    ]

    # one figure per class
    for c in range(n_classes):
        print(f"\n--- {class_names[c]} ---")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for spec_idx, (probs, name) in enumerate(zip(all_probs, specialist_names)):
            fpr, tpr, _ = roc_curve(Y_test_binarized[:, c], probs[:, c])
            class_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} AUC = {class_auc:.3f}",
                    color=colors[spec_idx])
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate", fontsize=label_fontsize)
        ax.set_ylabel("True Positive Rate", fontsize=label_fontsize)
        ax.legend(loc="lower right", fontsize=legend_fontsize)
        ax.grid(True)
        plt.tight_layout()

        if save_path is not None:
            base, ext = os.path.splitext(save_path)
            class_save_path = f"{base}_{class_names[c].replace(' ', '_')}{ext}"
            plt.savefig(class_save_path, format="svg", bbox_inches="tight")
            print(f"Saved to {class_save_path}")

        plt.show()
        plt.close(fig)
        
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix_from_scores(
    config, 
    T, 
    s,
    patch_size,
    max_lit,
    weighted,
    Y_true,
    specialist_name,
    dataset_name,
    save_specialist_name,
    base_save_dir="saved_confusion_matrices",
    figsize=(6, 5),
    cmap="Blues",
    rotate_xlabels=False,
    xlabels_rotation=30,
    annot_fontsize=12,
):
    label_names = [config.labels[i] for i in sorted(config.labels.keys())]
    file_path = (
        f"{config.SAVE_DIR_SCORES}/"
        f"{specialist_name}_{config.N_EPOCHS}_"
        f"{config.NUM_CLAUSES}_"
        f"{T}_"
        f"{s:.1f}_"
        f"{patch_size}_"
        f"{max_lit}_"
        f"{weighted}.txt"
    )
    class_sums = np.loadtxt(file_path, delimiter=",")
    Y_pred = np.argmax(class_sums, axis=1)
    acc = 100.0 * (Y_pred == Y_true).mean()
    print(f"Test Accuracy: {acc:.2f}%")
    
    cm = confusion_matrix(Y_true, Y_pred)
    
    # Normalize per row (true class) to get percentages
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    # Annotations: show percentage with one decimal
    annot = np.array([
        [f"{v:.1f}%" for v in row] for row in cm_percent
    ])
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm_percent,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=label_names,
        yticklabels=label_names,
        annot_kws={"size": annot_fontsize},
        vmin=0,
        vmax=100,
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    if rotate_xlabels:
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=xlabels_rotation,
            ha="right"
        )
    plt.tight_layout()
    save_dir = base_save_dir
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{dataset_name}_{save_specialist_name}.svg"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight", format="svg")
    print(f"Saved to {save_path}")
    plt.show()
    return acc, cm

import numpy as np
import matplotlib.pyplot as plt


def plot_clause_length_distribution(
    tm,
    bins=30,
    exclude_empty=True,
    show_plot=True,
):
    stats = {}

    for cls in range(tm.number_of_classes):

        clause_lengths = []
        for c in range(tm.number_of_clauses):
            length = tm.number_of_include_actions(cls, c)
            if not exclude_empty or length > 0:
                clause_lengths.append(length)

        if len(clause_lengths) == 0:
            stats[cls] = {
                "median": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "count": 0,
            }
            continue

        if show_plot:
            plt.hist(clause_lengths, bins=bins)
            plt.xlabel("Number of included literals per clause")
            plt.ylabel("Number of clauses")
            plt.title(f"Clause length distribution – class {cls}")
            plt.show()

        median = np.median(clause_lengths)
        mean = np.mean(clause_lengths)
        std = np.std(clause_lengths)

        print(
            f"Class {cls}: "
            f"median={median:.1f}, "
            f"mean={mean:.1f}, "
            f"std={std:.1f}"
        )

        stats[cls] = {
            "median": median,
            "mean": mean,
            "std": std,
            "count": len(clause_lengths),
        }

    return stats

def plot_literal_clause_frequency(tm, figsize=(14, 4)):
    literal_freq = tm.literal_clause_frequency()

    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(literal_freq)), literal_freq, width=1.0)
    plt.title("Literal usage frequency")
    plt.xlabel("Literal index")
    plt.ylabel("Frequency across clauses")
    plt.tight_layout()
    plt.show()

    return literal_freq



def get_clause_lengths_by_polarity(tm, target_class, exclude_empty=True):
    """
    Returns clause lengths separated by polarity for a given class.
    Positive polarity = first half of clauses
    Negative polarity = second half of clauses
    """
    n_clauses = tm.number_of_clauses
    half = n_clauses // 2

    pos_lengths = []
    neg_lengths = []

    for c in range(n_clauses):
        length = tm.number_of_include_actions(target_class, c)
        if exclude_empty and length == 0:
            continue
        if c < half:
            pos_lengths.append(length)
        else:
            neg_lengths.append(length)

    return np.array(pos_lengths), np.array(neg_lengths)


def plot_clause_length_boxplot_by_polarity(
    data_pos, data_neg, target_class,
    dataset_name='BreastMNIST', save_path=None,
    figsize=(10, 5),
    fontsize_labels=11,
    fontsize_legend=10,
    fontsize_ticks=11,
):
    specialists = list(data_pos.keys())
    n = len(specialists)

    print(f"\n--- Clause Length Observations: {dataset_name} | Class {target_class} ---")
    for s in specialists:
        n_pos = len(data_pos[s])
        n_neg = len(data_neg[s])
        print(f"  {s:<25} Positive: {n_pos:>6}  |  Negative: {n_neg:>6}")
    print("-" * 60)

    fig, ax = plt.subplots(figsize=figsize)
    positions_pos = np.arange(n) * 3
    positions_neg = np.arange(n) * 3 + 1
    bp_pos = ax.boxplot(
        [data_pos[s] for s in specialists],
        positions=positions_pos,
        widths=0.7,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='#3266ad',
                       markeredgecolor='white', markersize=5),
        boxprops=dict(facecolor='#b5d4f4', color='#3266ad'),
        medianprops=dict(color='#3266ad', linewidth=2),
        whiskerprops=dict(color='#3266ad', linestyle='--'),
        capprops=dict(color='#3266ad'),
        flierprops=dict(marker='o', markerfacecolor='#3266ad',
                        alpha=0.3, markersize=3),
    )
    bp_neg = ax.boxplot(
        [data_neg[s] for s in specialists],
        positions=positions_neg,
        widths=0.7,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='#d85a30',
                       markeredgecolor='white', markersize=5),
        boxprops=dict(facecolor='#f0997b', color='#d85a30'),
        medianprops=dict(color='#d85a30', linewidth=2),
        whiskerprops=dict(color='#d85a30', linestyle='--'),
        capprops=dict(color='#d85a30'),
        flierprops=dict(marker='o', markerfacecolor='#d85a30',
                        alpha=0.3, markersize=3),
    )
    tick_positions = (positions_pos + positions_neg) / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(specialists, fontsize=fontsize_ticks)
    ax.set_ylabel('Clause length (number of literals)', fontsize=fontsize_labels)
    ax.set_xlabel('Specialist', fontsize=fontsize_labels)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#b5d4f4', edgecolor='#3266ad', label='Positive clauses'),
        Patch(facecolor='#f0997b', edgecolor='#d85a30', label='Negative clauses'),
    ]
    ax.legend(handles=legend_elements, fontsize=fontsize_legend)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-1, positions_neg[-1] + 1)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', format='svg')
        print(f"Saved to {save_path}")
    plt.show()