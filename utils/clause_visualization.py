import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

def get_clause_pixel_contributions(
        X, X_orig, tm, image_index,
        resolution=1,
        n_channels=1,
        image_shape=(28, 28)):
    """
    Case 1: grayscale, no resolution  → X shape (N, H, W),     Z=1, R=1
    Case 2: grayscale, resolution     → X shape (N, H, W, R),  Z=1, R>1
    Case 3: RGB, no resolution        → X shape (N, H, W, 3),  Z=3, R=1
    Case 4: RGB, resolution           → X shape (N, H, W, 3R), Z=3, R>1

    Returns maps of shape (Z, R, H, W) — one per channel AND per level.
    """

    H, W = image_shape
    Z    = n_channels
    R    = resolution
    fpx  = Z * R

    n_cls  = tm.number_of_clauses
    half   = n_cls // 2
    ph, pw = tm.patch_dim[0], tm.patch_dim[1]
    grid_rows = H - ph + 1
    grid_cols = W - pw + 1

    n_y_pos_bits = H - ph
    n_x_pos_bits = W - pw
    patch_start  = n_y_pos_bits + n_x_pos_bits
    n_pixels     = ph * pw
    n_features   = patch_start + n_pixels * fpx

    # raw image for display
    if X_orig.ndim == 3:
        raw = X_orig[image_index].reshape((H, W)).astype(np.float32)
    else:
        raw = X_orig[image_index].reshape((H, W, Z)).astype(np.float32)

    pred_class  = int(tm.predict(X[image_index:image_index + 1])[0])
    Xt_patch    = tm.transform_patchwise(X[image_index:image_index + 1])
    cls_patch   = Xt_patch[0, pred_class * n_cls:(pred_class + 1) * n_cls, :]
    raw_weights = tm.weight_banks[pred_class].get_weights()

    # (Z, R, H, W) — separate per channel AND per level
    pixel_maps_pos = np.zeros((Z, R, H, W), dtype=np.float32)
    pixel_maps_neg = np.zeros((Z, R, H, W), dtype=np.float32)

    for patch_idx in range(grid_rows * grid_cols):
        pr = patch_idx // grid_cols
        pc = patch_idx  % grid_cols

        for ci in range(n_cls):
            if cls_patch[ci, patch_idx] == 0:
                continue

            w               = raw_weights[ci]
            clause_polarity = 0 if w > 0 else 1

            for pixel_in_patch in range(n_pixels):
                orig_row = pr + (pixel_in_patch // pw)
                orig_col = pc + (pixel_in_patch  % pw)

                for z in range(Z):
                    for lvl in range(R):
                        inner   = pixel_in_patch * fpx + z * R + lvl
                        lit_pos = patch_start + inner
                        lit_neg = n_features + patch_start + inner

                        if tm.get_ta_action(ci % half, lit_pos,
                                            the_class=pred_class,
                                            polarity=clause_polarity):
                            if w > 0:
                                pixel_maps_pos[z, lvl, orig_row, orig_col] += abs(w)
                            else:
                                pixel_maps_neg[z, lvl, orig_row, orig_col] += abs(w)

                        if tm.get_ta_action(ci % half, lit_neg,
                                            the_class=pred_class,
                                            polarity=clause_polarity):
                            if w > 0:
                                pixel_maps_pos[z, lvl, orig_row, orig_col] -= abs(w)
                            else:
                                pixel_maps_neg[z, lvl, orig_row, orig_col] -= abs(w)

    # booleanized images: (Z, R, H, W)
    ch_level_images = np.zeros((Z, R, H, W), dtype=np.float32)
    for z in range(Z):
        for lvl in range(R):
            if X.ndim == 3:               # Case 1: (N, H, W)
                ch_level_images[z, lvl] = X[image_index].astype(np.float32)
            elif X.ndim == 4 and Z == 1:  # Case 2: (N, H, W, R)
                ch_level_images[z, lvl] = X[image_index, :, :, lvl].astype(np.float32)
            else:                         # Case 3: (N, H, W, Z)  z*1+0=z
                                          # Case 4: (N, H, W, Z*R) z*R+lvl
                ch_level_images[z, lvl] = X[image_index, :, :, z * R + lvl].astype(np.float32)

    return pixel_maps_pos, pixel_maps_neg, ch_level_images, pred_class, raw


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _draw_heatmap(ax, data, vmin, vmax, title, fontsize=12):
    ax.imshow(data, cmap="icefire", vmin=vmin, vmax=vmax,
              aspect='equal', interpolation='nearest')
    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")


def _show_raw(ax, raw, n_channels, title):
    if n_channels == 1:
        ax.imshow(raw, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(np.clip(raw / 255.0, 0, 1))
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def _show_binary(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#e8e8e8')  # lighter than 'lightgray'
        spine.set_linewidth(0.8)
        spine.set_alpha(0.5)


def _heatmap(ax, data, vmax, title):
    ax.imshow(data, cmap="icefire", vmin=-vmax, vmax=vmax,
              aspect='equal', interpolation='bilinear')
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    


# ---------------------------------------------------------------------------
# 1. Aggregated heatmap — sum over ALL channels and ALL levels → 1 row
# ---------------------------------------------------------------------------

def visualize_aggregated_heatmap(
        X, X_orig, tm, image_index=0,
        resolution=1, n_channels=1, image_shape=(28, 28),
        show_diff=True, save_path=None):

    pixel_maps_pos, pixel_maps_neg, ch_level_images, pred_class, raw = \
        get_clause_pixel_contributions(X, X_orig, tm, image_index,
                                       resolution, n_channels, image_shape)

    agg_pos = pixel_maps_pos.sum(axis=(0, 1))
    agg_neg = pixel_maps_neg.sum(axis=(0, 1))
    agg     = agg_pos - agg_neg
    vmax    = max(np.abs(agg_pos).max(), np.abs(agg_neg).max(),
                  np.abs(agg).max(), 1e-6)

    heatmaps = [
        (agg_pos, "Aggregation of Positive Clauses"),
        (agg_neg, "Aggregation of Negative Clauses"),
    ]
    if show_diff:
        heatmaps.append((agg, "Aggregation of Positive - Negative Clauses"))

    n_cols = 1 + len(heatmaps) + 1
    width_ratios = [1] * (n_cols - 1) + [0.05]

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * (n_cols - 1), 5),
                             gridspec_kw={"width_ratios": width_ratios,
                                          "wspace": 0.05})

    # fig.suptitle(f"Aggregated Heatmap  --  sample {image_index}  |  pred: {pred_class}",
    #              fontsize=12)
    print(f"Visualizing sample {image_index}  |  predicted class: {pred_class}")
    _show_raw(axes[0], raw, n_channels, f"Original")

    for i, (data, title) in enumerate(heatmaps):
        axes[i + 1].imshow(data, cmap="icefire", vmin=-vmax, vmax=vmax,
                       aspect='equal')
        axes[i + 1].set_title(title, fontsize=12)
        axes[i + 1].axis("off")

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm = cm.ScalarMappable(cmap="icefire", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=axes[-1])

    plt.subplots_adjust(wspace=0.05)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", format="svg")
        print(f"Saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 2. Per-channel heatmap — sum over levels → Z rows
# ---------------------------------------------------------------------------

def visualize_per_channel_heatmaps(
        X, X_orig, tm, image_index=0,
        resolution=1, n_channels=1, image_shape=(28, 28),
        channel_names=None, show_diff=True, save_path=None):
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    Z = n_channels
    if channel_names is None:
        channel_names = [f"Ch{z}" for z in range(Z)]
    pixel_maps_pos, pixel_maps_neg, ch_level_images, pred_class, raw = \
        get_clause_pixel_contributions(X, X_orig, tm, image_index,
                                       resolution, n_channels, image_shape)
    ch_pos = pixel_maps_pos.sum(axis=1)
    ch_neg = pixel_maps_neg.sum(axis=1)
    n_rows = Z
    n_heatmap_cols = 3 if show_diff else 2
    print(f"Visualizing sample {image_index}  |  predicted class: {pred_class}")
    if Z == 1:
        # ---- SINGLE CHANNEL: flat row layout ----
        n_cols = 1 + 1 + n_heatmap_cols + 1
        width_ratios = [1] * (n_cols - 1) + [0.05]
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(4 * (n_cols - 1), 3.5),
            gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05}
        )
        pos  = ch_pos[0]
        neg  = ch_neg[0]
        diff = pos - neg
        vmax = max(np.abs(pos).max(), np.abs(neg).max(), np.abs(diff).max(), 1e-6)
        ch_img = ch_level_images[0].max(axis=0)
        _show_raw(axes[0], raw, n_channels, "Original")
        axes[0].title.set_fontsize(12)
        _show_binary(axes[1], ch_img, channel_names[0])
        axes[1].title.set_fontsize(12)
        heatmaps = [
            (axes[2], pos, "Positive clauses"),
            (axes[3], neg, "Negative clauses"),
        ]
        if show_diff:
            heatmaps.append((axes[4], diff, "Pos - Neg"))
        for ax, data, title in heatmaps:
            ax.imshow(data, cmap="icefire", vmin=-vmax, vmax=vmax, aspect='equal')
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        sm = cm.ScalarMappable(cmap="icefire", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=axes[-1])
    else:
        # ---- MULTI CHANNEL: same layout as per_channel_level ----
        fig = plt.figure(
            figsize=(2.5 * (1 + n_heatmap_cols + 1), 2.5 * n_rows + 1.5)
        )
        outer = gridspec.GridSpec(
            2, 3,
            height_ratios=[0.79, n_rows],
            width_ratios=[1, 1, 1],
            hspace=0.12,
            wspace=-0.8
        )
        # ---- ORIGINAL IMAGE ----
        ax_orig = fig.add_subplot(outer[0, 1])
        _show_raw(ax_orig, raw, n_channels, "Original")
        ax_orig.title.set_fontsize(12)
        fig.add_subplot(outer[0, 0]).axis("off")
        fig.add_subplot(outer[0, 2]).axis("off")
        # ---- INNER GRID ----
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows,
            1 + n_heatmap_cols + 1,
            subplot_spec=outer[1, :],
            width_ratios=[1] + [1] * n_heatmap_cols + [0.05],
            wspace=0.05,
            hspace=0.25
        )
        for z in range(Z):
            pos  = ch_pos[z]
            neg  = ch_neg[z]
            diff = pos - neg
            vmax = max(np.abs(pos).max(), np.abs(neg).max(), np.abs(diff).max(), 1e-6)
            ch_img = ch_level_images[z].max(axis=0)
            # ---- BINARY IMAGE ----
            ax_bin = fig.add_subplot(inner[z, 0])
            _show_binary(ax_bin, ch_img, channel_names[z])
            ax_bin.title.set_fontsize(11)
            ax_bin.set_aspect('equal')
            # ---- HEATMAPS ----
            heatmaps = [
                (pos, "Positive clauses"),
                (neg, "Negative clauses"),
            ]
            if show_diff:
                heatmaps.append((diff, "Pos - Neg"))
            for col_idx, (data, title) in enumerate(heatmaps):
                ax = fig.add_subplot(inner[z, 1 + col_idx])
                ax.imshow(data, cmap="icefire", vmin=-vmax, vmax=vmax, aspect='equal')
                ax.set_title(title, fontsize=11)
                ax.axis("off")
            # ---- COLORBAR ----
            ax_cb = fig.add_subplot(inner[z, -1])
            norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
            sm = cm.ScalarMappable(cmap="icefire", norm=norm)
            sm.set_array([])
            fig.colorbar(sm, cax=ax_cb)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", format="svg")
        print(f"Saved to {save_path}")
    plt.show()
# ---------------------------------------------------------------------------
# 3. Per-channel-level heatmap — one row per (channel, level) → Z*R rows
# ---------------------------------------------------------------------------
def _draw_channel_figure(
        z, R, n_heatmap_cols, pixel_maps_pos, pixel_maps_neg,
        ch_level_images, raw, n_channels, pred_class,
        image_index, use_channel_names, channel_names):
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n_rows = R
    fig = plt.figure(
        figsize=(2.5 * (1 + n_heatmap_cols + 1), 2.5 * n_rows + 1.5)
    )
    outer = gridspec.GridSpec(
        2, 3,
        height_ratios=[0.79, n_rows],
        width_ratios=[1, 1, 1],
        hspace=0.05,
        wspace=-0.8
    )
    ax_orig = fig.add_subplot(outer[0, 1])
    _show_raw(ax_orig, raw, n_channels, "Original")
    ax_orig.title.set_fontsize(12)
    fig.add_subplot(outer[0, 0]).axis("off")
    fig.add_subplot(outer[0, 2]).axis("off")

    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows,
        1 + n_heatmap_cols + 1,
        subplot_spec=outer[1, :],
        width_ratios=[1] + [1] * n_heatmap_cols + [0.05],
        wspace=0.05,
        hspace=0.25
    )
    for lvl in range(R):
        pos  = pixel_maps_pos[z, lvl]
        neg  = pixel_maps_neg[z, lvl]
        diff = pos - neg
        vmax = max(
            np.abs(pos).max(), np.abs(neg).max(),
            np.abs(diff).max(), 1e-6
        )
        if use_channel_names:
            label = f"{channel_names[z]} Level {lvl + 1}" if R > 1 else f"{channel_names[z]}"
        else:
            label = f"Level {lvl + 1}" if R > 1 else ""

        ax_bin = fig.add_subplot(inner[lvl, 0])
        _show_binary(ax_bin, ch_level_images[z, lvl], label)
        ax_bin.title.set_fontsize(12)
        ax_bin.set_aspect('equal')

        heatmaps = [(pos, "Positive clauses"), (neg, "Negative clauses")]
        if n_heatmap_cols == 3:
            heatmaps.append((diff, "Pos - Neg"))

        for col_idx, (data, title) in enumerate(heatmaps):
            ax = fig.add_subplot(inner[lvl, 1 + col_idx])
            ax.imshow(data, cmap="icefire", vmin=-vmax, vmax=vmax, aspect='equal')
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        ax_cb = fig.add_subplot(inner[lvl, -1])
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        sm = cm.ScalarMappable(cmap="icefire", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=ax_cb)

    return fig


def visualize_per_channel_level_heatmaps(
        X, X_orig, tm, image_index=0,
        resolution=1, n_channels=1, image_shape=(28, 28),
        channel_names=None, show_diff=True, save_path=None):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    Z = n_channels
    R = resolution
    use_channel_names = channel_names is not None
    pixel_maps_pos, pixel_maps_neg, ch_level_images, pred_class, raw = \
        get_clause_pixel_contributions(
            X, X_orig, tm, image_index,
            resolution, n_channels, image_shape
        )
    n_heatmap_cols = 3 if show_diff else 2
    print(f"Visualizing sample {image_index}  |  predicted class: {pred_class}")

    if Z == 1:
        # single figure, single save path
        fig = _draw_channel_figure(
            0, R, n_heatmap_cols, pixel_maps_pos, pixel_maps_neg,
            ch_level_images, raw, n_channels, pred_class,
            image_index, use_channel_names, channel_names
        )
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", format="svg")
            print(f"Saved to {save_path}")
        plt.show()
    else:
        # one figure per channel, save_path must be a list of length Z
        if save_path is not None and not isinstance(save_path, (list, tuple)):
            raise ValueError(
                f"n_channels={Z} > 1: save_path must be a list of {Z} paths, one per channel."
            )
        for z in range(Z):
            ch_label = channel_names[z] if use_channel_names else f"Ch{z}"
            print(f"  Channel {z} ({ch_label})")
            fig = _draw_channel_figure(
                z, R, n_heatmap_cols, pixel_maps_pos, pixel_maps_neg,
                ch_level_images, raw, n_channels, pred_class,
                image_index, use_channel_names, channel_names
            )
            if save_path is not None:
                p = save_path[z]
                os.makedirs(os.path.dirname(p), exist_ok=True)
                fig.savefig(p, dpi=150, bbox_inches="tight", format="svg")
                print(f"  Saved to {p}")
            plt.show()
            plt.close(fig)
    
    

def visualize_average_heatmap(
        X, X_orig, Y, tm,
        resolution=1, n_channels=1, image_shape=(28, 28),
        target_class=None,
        max_samples=None,
        show_diff=True,
        show_sample=True,
        save_path=None, bigger_font=False):

    H, W = image_shape
    Z    = n_channels
    R    = resolution

    Y_pred = np.array([tm.predict(X[i:i+1])[0] for i in range(len(X))])

    correct_idx = np.where(Y_pred == Y)[0]

    if target_class is not None:
        correct_idx = correct_idx[Y[correct_idx] == target_class]

    if max_samples is not None:
        correct_idx = correct_idx[:max_samples]

    n = len(correct_idx)
    if n == 0:
        print("No correctly predicted samples found.")
        return

    print(f"Averaging over {n} correctly predicted samples"
          + (f" of class {target_class}" if target_class is not None else ""))

    agg_pos_sum = np.zeros((H, W), dtype=np.float64)
    agg_neg_sum = np.zeros((H, W), dtype=np.float64)

    for i, idx in enumerate(correct_idx):
        pixel_maps_pos, pixel_maps_neg, _, _, _ = get_clause_pixel_contributions(
            X, X_orig, tm, int(idx), resolution, n_channels, image_shape)
        agg_pos_sum += pixel_maps_pos.sum(axis=(0, 1))
        agg_neg_sum += pixel_maps_neg.sum(axis=(0, 1))
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{n}")

    avg_pos  = agg_pos_sum / n
    avg_neg  = agg_neg_sum / n
    avg_diff = avg_pos - avg_neg

    vmax = max(np.abs(avg_pos).max(), np.abs(avg_neg).max(),
               np.abs(avg_diff).max(), 1e-6)

    # build heatmap list
    heatmaps = [
        (avg_pos, "Positive"),
        (avg_neg, "Negative"),
    ]
    if show_diff:
        heatmaps.append((avg_diff, "Pos - Neg"))

    # build axes list
    n_heatmap_cols = len(heatmaps)
    n_cols = (1 if show_sample else 0) + n_heatmap_cols + 1
    width_ratios = [1] * (n_cols - 1) + [0.05]

    title_cls = f"class {target_class}" if target_class is not None else "all classes"

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * (n_cols - 1), 5),
                             gridspec_kw={"width_ratios": width_ratios,
                                          "wspace": 0.05})

    # fig.suptitle(f"Average Heatmap  --  {title_cls}  |  n={n} correct predictions",
    #              fontsize=12)

    # offset for heatmap axes depends on whether sample is shown
    offset = 0
    if show_sample:
        if X_orig.ndim == 3:
            raw = X_orig[correct_idx[0]].reshape((H, W)).astype(np.float32)
        else:
            raw = X_orig[correct_idx[0]].reshape((H, W, Z)).astype(np.float32)
        _show_raw(axes[0], raw, n_channels, f"Sample (class {Y[correct_idx[0]]})")
        offset = 1

    for i, (data, title) in enumerate(heatmaps):
        axes[i + offset].imshow(data, cmap="icefire", vmin=-vmax, vmax=vmax, aspect='equal')
        axes[i + offset].set_title(title, fontsize=14)
        axes[i + offset].axis("off")

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm = cm.ScalarMappable(cmap="icefire", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=axes[-1])

    plt.subplots_adjust(wspace=0.05)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", format="svg")
        print(f"Saved to {save_path}")

    plt.show()

    return avg_pos, avg_neg, avg_diff
 


def number_of_literals_in_clause(tm, Y_test):
    # calculate number of classes
    n_classes = len(np.unique(Y_test))
    print(f"Number of classes: {n_classes}")
    for i in range(n_classes):
        print(f"Class {i} has {tm.clause_banks[i].number_of_literals} literals.")



def get_correct_sample_index(tm, X, Y, target_class, random=False, seed=42):
    Y_pred = np.array([tm.predict(X[i:i+1])[0] for i in range(len(X))])
    correct_idx = np.where((Y == target_class) & (Y_pred == target_class))[0]
    
    if len(correct_idx) == 0:
        print(f"No correctly predicted samples found for class {target_class}")
        return None
    
    return int(np.random.choice(correct_idx))




def save_heatmap_images(agg_pos, agg_neg, agg, raw, n_channels, image_index, pred_class, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    vmax = max(np.abs(agg_pos).max(), np.abs(agg_neg).max(),
               np.abs(agg).max(), 1e-6)
    
    # save heatmaps using _heatmap
    for name, data in [("pos", agg_pos), ("neg", agg_neg), ("diff", agg)]:
        fig, ax = plt.subplots(figsize=(3, 3))
        _heatmap(ax, data, vmax, title="")
        ax.set_title("")
        plt.tight_layout(pad=0)
        path = os.path.join(save_dir, f"sample_{image_index}_class_{pred_class}_{name}.svg")
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved {path}")
    
    # save raw image using _show_raw
    fig, ax = plt.subplots(figsize=(3, 3))
    _show_raw(ax, raw, n_channels, title="")
    plt.tight_layout(pad=0)
    path = os.path.join(save_dir, f"sample_{image_index}_class_{pred_class}_raw.svg")
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved {path}")




def load_raw_maps_and_plot_heatmap(save_dir, specialist, target_class, 
                                    save_path=None,
                                    title_pos=None, title_neg=None, bigger_font=False):
    """
    Quick test to verify saved heatmaps load and display correctly.
    """
    avg_pos = np.load(f"{save_dir}/global/{specialist}/{specialist}_c{target_class}_pos.npy")
    avg_neg = np.load(f"{save_dir}/global/{specialist}/{specialist}_c{target_class}_neg.npy")
    vmax = max(np.abs(avg_pos).max(), np.abs(avg_neg).max(), 1e-6)
    import matplotlib.colors as mcolors

    title_pos = title_pos if title_pos is not None else f"Positive — {specialist} — class {target_class}"
    title_neg = title_neg if title_neg is not None else f"Negative — {specialist} — class {target_class}"

    width_ratios = [1, 1, 0.05]
    fig, axes = plt.subplots(1, 3, figsize=(10, 5),
                             gridspec_kw={"width_ratios": width_ratios,
                                          "wspace": 0.05})
    axes[0].imshow(avg_pos, cmap="icefire", vmin=-vmax, vmax=vmax, aspect='equal')
    axes[0].set_title(title_pos, fontsize=13 if not bigger_font else 20)
    axes[0].axis("off")

    axes[1].imshow(avg_neg, cmap="icefire", vmin=-vmax, vmax=vmax, aspect='equal')
    axes[1].set_title(title_neg, fontsize=13 if not bigger_font else 20)
    axes[1].axis("off")
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm = cm.ScalarMappable(cmap="icefire", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=axes[-1])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", format="svg")
        print(f"Saved to {save_path}")

    plt.show()
    print(f"pos shape: {avg_pos.shape}, min: {avg_pos.min():.2f}, max: {avg_pos.max():.2f}")
    print(f"neg shape: {avg_neg.shape}, min: {avg_neg.min():.2f}, max: {avg_neg.max():.2f}")
    

def build_save_path(
        base_dir,
        dataset,      # e.g. "breast"
        specialist,   # e.g. "canny", "adaptive_gaussian"
        heatmap_type, # "lh" or "gh"
        target_class, # e.g. 0 or 1
        heatmap_level=None,  # "aggr", "chan", "chanlvl" — optional
        sample_idx=None,     # only for local heatmaps
        channel=None,        # only for chan / chanlvl
        level=None           # only for chanlvl
):
    parts = [dataset, specialist, heatmap_type]

    

    parts.append(f"c{target_class}")

    if sample_idx is not None:
        parts.append(f"s{sample_idx}")
    if heatmap_level is not None:
        parts.append(heatmap_level)
    if channel is not None:
        parts.append(f"ch{channel}")
    if level is not None:
        parts.append(f"l{level}")

    filename = "_".join(parts) + ".svg"
    return os.path.join(base_dir, filename)