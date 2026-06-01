import numpy as np
import cv2
import math
from tqdm import tqdm


def perform_color_thermometers_booleanization(
    X_train_org,
    X_val_org,
    X_test_org,
    Y_train,
    Y_val,
    Y_test,
    resolution,
):
    def encode(X_org):
        if X_org.ndim == 3:  # Grayscale (N, H, W) → (N, H, W, R)
            N, H, W = X_org.shape
            X_encoded = np.empty((N, H, W, resolution), dtype=np.uint8)
            for z in range(resolution):
                threshold = (z + 1) * 255 / (resolution + 1)
                X_encoded[..., z] = X_org >= threshold
            return X_encoded  # (N, H, W, R)

        elif X_org.ndim == 4:  # Multi-channel (N, H, W, C) → (N, H, W, C*R)
            N, H, W, C = X_org.shape
            X_encoded = np.empty((N, H, W, C, resolution), dtype=np.uint8)
            for z in range(resolution):
                threshold = (z + 1) * 255 / (resolution + 1)
                X_encoded[..., z] = X_org >= threshold
            return X_encoded.reshape(N, H, W, C * resolution)  # (N, H, W, C*R)

        else:
            raise ValueError("Input must have shape (N,H,W) or (N,H,W,C)")

    X_train = encode(X_train_org)
    X_val   = encode(X_val_org)
    X_test  = encode(X_test_org)

    Y_train = Y_train.reshape(-1)
    Y_val   = Y_val.reshape(-1)
    Y_test  = Y_test.reshape(-1)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test




def perform_canny_edge_detection_booleanization(
    X_train_org,
    X_val_org,
    X_test_org,
    Y_train,
    Y_val,
    Y_test,
    blur_kernel=(3, 3),
    canny_threshold1=100,
    canny_threshold2=200,
):
    """
    Copy datasets, reshape labels, and apply Gaussian blur + Canny edge detection.

    Supports:
        (N, H, W)      grayscale
        (N, H, W, C)   multi-channel

    Returns:
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test
    """

    # Copy inputs
    X_train = np.copy(X_train_org)
    X_test  = np.copy(X_test_org)
    X_val   = np.copy(X_val_org)

    # Reshape labels
    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_test = Y_test.reshape(Y_test.shape[0])
    Y_val = Y_val.reshape(Y_val.shape[0])

    def process_split(X):

        if X.ndim == 3:  # Grayscale
            N, H, W = X.shape
            X_edges = np.empty((N, H, W), dtype=np.uint8)

            for i in range(N):
                blur = cv2.GaussianBlur(X[i], blur_kernel, 0)
                edges = cv2.Canny(blur, canny_threshold1, canny_threshold2)
                X_edges[i] = (edges != 0).astype(np.uint8)

        elif X.ndim == 4:  # Multi-channel
            N, H, W, C = X.shape
            X_edges = np.empty((N, H, W, C), dtype=np.uint8)

            for i in range(N):
                for c in range(C):
                    blur = cv2.GaussianBlur(X[i, :, :, c], blur_kernel, 0)
                    edges = cv2.Canny(blur, canny_threshold1, canny_threshold2)
                    X_edges[i, :, :, c] = (edges != 0).astype(np.uint8)

        else:
            raise ValueError("Input must have shape (N,H,W) or (N,H,W,C)")

        return X_edges

    X_train = process_split(X_train)
    X_test  = process_split(X_test)
    X_val   = process_split(X_val)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test





def perform_adaptive_gaussian_thresholding_booleanization(
    X_train_org,
    X_val_org,
    X_test_org,
    Y_train,
    Y_val,
    Y_test,
    max_value=1,
    block_size=11,
    C=2,
):
    """
    Apply adaptive Gaussian thresholding to grayscale or 3-channel images.

    Supports:
        (N, H, W)
        (N, H, W, 3)

    Returns:
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test
    """

    # Reshape labels
    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_val   = Y_val.reshape(Y_val.shape[0])
    Y_test  = Y_test.reshape(Y_test.shape[0])

    # Cast images
    X_train = X_train_org.astype(np.uint8)
    X_test  = X_test_org.astype(np.uint8)
    X_val   = X_val_org.astype(np.uint8)

    def process_split(X):

        if X.ndim == 3:  # Grayscale
            N, H, W = X.shape
            X_out = np.empty((N, H, W), dtype=np.uint8)

            for i in range(N):
                X_out[i] = cv2.adaptiveThreshold(
                    X[i],
                    max_value,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    C,
                )

        elif X.ndim == 4:  # Multi-channel (e.g., RGB)
            N, H, W, C_channels = X.shape
            X_out = np.empty((N, H, W, C_channels), dtype=np.uint8)

            for i in range(N):
                for c in range(C_channels):
                    X_out[i, :, :, c] = cv2.adaptiveThreshold(
                        X[i, :, :, c],
                        max_value,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        block_size,
                        C,
                    )

        else:
            raise ValueError("Input must have shape (N,H,W) or (N,H,W,3)")

        return X_out

    X_train = process_split(X_train)
    X_test  = process_split(X_test)
    X_val   = process_split(X_val)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test




def multi_level_threshold(image, resolution=8):
    """
    Compute multi-level thresholds for a single grayscale image.
    """
    img = np.array(image)
    a, b = 0, 255
    n = resolution
    k = 0.7
    T = []

    def multiThresh(img, a, b):
        if a > b:
            return -1, -1

        mask = (img >= a) & (img <= b)
        s = np.sum(mask)
        if s == 0:
            return -1, -1
        m = np.sum(img * mask) / s
        return m, s

    for i in range(int(n / 2 - 1)):
        mask = (img >= a) & (img <= b)
        region = img * mask

        s = np.sum(mask)
        if s == 0:
            break

        mu = np.sum(region) / s
        sigma = math.sqrt(np.sum(((region - mu) * mask) ** 2) / s)

        T1 = mu - k * sigma
        T2 = mu + k * sigma

        x, _ = multiThresh(img, a, T1)
        w, _ = multiThresh(img, T2, b)

        T.extend([x, w])

        a = T1 + 1
        b = T2 - 1
        k *= (i + 1)

    # Final split
    x, _ = multiThresh(img, a, mu)
    w, _ = multiThresh(img, mu + 1, b)
    T.extend([x, w])

    T = [t for t in T if t != -1]
    T.sort()
    return T


def perform_adaptive_color_thermometer_booleanization(
    X_train_org,
    X_val_org,
    X_test_org,
    Y_train,
    Y_val,
    Y_test,
    resolution=8,
):
    """
    Apply multi-level thresholding to grayscale or multi-channel images.
    Also reshapes labels.

    Supports:
        (N, H, W)
        (N, H, W, C)

    Returns:
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test
    """

    # 🔹 Reshape labels (as requested)
    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_val   = Y_val.reshape(Y_val.shape[0])
    Y_test  = Y_test.reshape(Y_test.shape[0])

    def process_split(image_array):

        if image_array.ndim == 3:  # Grayscale
            N, H, W = image_array.shape
            output = np.empty((N, H, W, resolution), dtype=np.uint8)

            for i in tqdm(range(N)):
                thresholds = multi_level_threshold(
                    image_array[i], resolution
                )

                for z, threshold in enumerate(thresholds):
                    output[i, :, :, z] = (
                        image_array[i] >= threshold
                    )

        elif image_array.ndim == 4:  # Multi-channel
            N, H, W, C = image_array.shape
            output = np.empty((N, H, W, C, resolution), dtype=np.uint8)

            for i in tqdm(range(N)):
                for c in range(C):
                    thresholds = multi_level_threshold(
                        image_array[i, :, :, c], resolution
                    )

                    for z, threshold in enumerate(thresholds):
                        output[i, :, :, c, z] = (
                            image_array[i, :, :, c] >= threshold
                        )

        else:
            raise ValueError("Input must have shape (N,H,W) or (N,H,W,C)")

        return output

    X_train = process_split(X_train_org)
    X_val   = process_split(X_val_org)
    X_test  = process_split(X_test_org)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test





def perform_adaptive_mean_thresholding_booleanization(
    X_train_org,
    X_val_org,
    X_test_org,
    Y_train,
    Y_val,
    Y_test,
    max_value=1,
    block_size=11,
    C=2,
):
    """
    Apply adaptive mean thresholding to train, val, and test splits.

    Supports:
        (N, H, W)
        (N, H, W, C)

    Returns:
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test
    """

    # Copy + cast
    X_train = np.copy(X_train_org).astype(np.uint8)
    X_val   = np.copy(X_val_org).astype(np.uint8)
    X_test  = np.copy(X_test_org).astype(np.uint8)

    # Reshape labels
    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_val   = Y_val.reshape(Y_val.shape[0])
    Y_test  = Y_test.reshape(Y_test.shape[0])

    print(len(Y_train))

    def process_split(X):

        if X.ndim == 3:  # Grayscale
            N, H, W = X.shape
            X_out = np.empty((N, H, W), dtype=np.uint8)

            for i in range(N):
                X_out[i] = cv2.adaptiveThreshold(
                    X[i],
                    max_value,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    C,
                )

        elif X.ndim == 4:  # Multi-channel
            N, H, W, C_channels = X.shape
            X_out = np.empty((N, H, W, C_channels), dtype=np.uint8)

            for i in range(N):
                for j in range(C_channels):
                    X_out[i, :, :, j] = cv2.adaptiveThreshold(
                        X[i, :, :, j],
                        max_value,
                        cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY,
                        block_size,
                        C,
                    )

        else:
            raise ValueError("Input must have shape (N,H,W) or (N,H,W,C)")

        return X_out

    X_train = process_split(X_train)
    X_val   = process_split(X_val)
    X_test  = process_split(X_test)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def perform_otsu_thresholding_booleanization(
    X_train_org,
    X_val_org,
    X_test_org,
    Y_train,
    Y_val,
    Y_test,
):
    """
    Apply Otsu thresholding to train, val, and test splits.

    Supports:
        (N, H, W)
        (N, H, W, C)

    Returns:
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test
    """

    # Copy + cast
    X_train = np.copy(X_train_org).astype(np.uint8)
    X_val   = np.copy(X_val_org).astype(np.uint8)
    X_test  = np.copy(X_test_org).astype(np.uint8)

    # Reshape labels
    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_val   = Y_val.reshape(Y_val.shape[0])
    Y_test  = Y_test.reshape(Y_test.shape[0])

    def process_split(X):

        if X.ndim == 3:  # Grayscale
            N, H, W = X.shape
            X_out = np.empty((N, H, W), dtype=np.uint8)

            for i in range(N):
                _, X_out[i] = cv2.threshold(
                    X[i],
                    0,
                    1,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )

        elif X.ndim == 4:  # Multi-channel
            N, H, W, C_channels = X.shape
            X_out = np.empty((N, H, W, C_channels), dtype=np.uint8)

            for i in range(N):
                for j in range(C_channels):
                    _, X_out[i, :, :, j] = cv2.threshold(
                        X[i, :, :, j],
                        0,
                        1,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                    )

        else:
            raise ValueError("Input must have shape (N,H,W) or (N,H,W,C)")

        return X_out

    X_train = process_split(X_train)
    X_val   = process_split(X_val)
    X_test  = process_split(X_test)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


import numpy as np


def perform_hog_booleanization(
    hog,
    X_train_org,
    X_val_org,
    X_test_org,
    Y_train,
    Y_val,
    Y_test,
    threshold=0.1,
):
    """
    Compute binarized HOG features for train, val, and test splits.

    Supports:
        (N, H, W)
        (N, H, W, 3)

    Returns:
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test
    """

    # --- Reshape labels ---
    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_val   = Y_val.reshape(Y_val.shape[0])
    Y_test  = Y_test.reshape(Y_test.shape[0])

    def process_split(X):

        # Determine feature length using first sample
        if X.ndim == 3:  # Grayscale
            fd = hog.compute(X[0])
            feature_length = fd.shape[0]

            X_out = np.empty((X.shape[0], feature_length), dtype=np.uint32)

            for i in range(X.shape[0]):
                fd = hog.compute(X[i])
                X_out[i] = (fd >= threshold).astype(np.uint32)

        elif X.ndim == 4:  # Multi-channel (e.g. RGB)
            # Compute feature length per channel
            fd = hog.compute(X[0, :, :, 0])
            feature_length = fd.shape[0]
            total_features = feature_length * X.shape[3]

            X_out = np.empty((X.shape[0], total_features), dtype=np.uint32)

            for i in range(X.shape[0]):
                feature_vector = []
                for c in range(X.shape[3]):
                    fd = hog.compute(X[i, :, :, c])
                    feature_vector.append(fd)

                feature_vector = np.concatenate(feature_vector)
                X_out[i] = (feature_vector >= threshold).astype(np.uint32)

        else:
            raise ValueError("Input must have shape (N,H,W) or (N,H,W,3)")

        return X_out

    X_train = process_split(X_train_org)
    X_val   = process_split(X_val_org)
    X_test  = process_split(X_test_org)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

