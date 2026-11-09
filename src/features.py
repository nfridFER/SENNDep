import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy as shannon_entropy
from scipy.signal import coherence


def _band_list():
    return [
        ("theta", (4, 8)),
        ("alpha", (8, 12)),
        ("beta", (12, 30)),
        ("delta", (0.5, 4)),
        ("gamma", (30, 45)),
    ]


def make_epoch_feature_names(ch_names):
    """
    Must match extract_epoch_features() exactly (ordering matters).
    """
    names = []

    # A) Relative bandpowers per channel (band-major)
    for band_name, _ in _band_list():
        for ch in ch_names:
            names.append(f"relpow_{band_name}[{ch}]")

    # B) Channel-wise stats + nonlinearities (channel-major)
    ch_metrics = [
        "mean",
        "std",
        "skew",
        "kurtosis",
        "sampen",
        "higuchi_fd",
        "c0_complexity",
        "shannon_H",
        "approx_entropy",
        "renyi_entropy",
    ]
    for ch in ch_names:
        for m in ch_metrics:
            names.append(f"{m}[{ch}]")

    # C) Cross-channel ratios (epoch-wise averages)
    names.append("ratio_mean_theta_over_alpha")
    names.append("ratio_mean_theta_over_beta")

    # D) Alpha coherence pairs (i<j)
    for i in range(len(ch_names)):
        for j in range(i + 1, len(ch_names)):
            names.append(f"coh_alpha[{ch_names[i]}-{ch_names[j]}]")

    # E) F3–F4 alpha asymmetry
    names.append("alpha_asym[F3-F4]")

    return names





def compute_coherence_matrix(epoch_data, sfreq, band=(8, 12)):
    """Flattened upper-triangle coherence values for all channel pairs."""
    n_channels = epoch_data.shape[0]
    coh_values = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            f, Cxy = coherence(epoch_data[i], epoch_data[j], fs=sfreq, nperseg=256)
            band_mask = (f >= band[0]) & (f <= band[1])
            coh_values.append(np.mean(Cxy[band_mask]))
    return np.array(coh_values)


def c0_complexity(signal):
    """C0 complexity proxy via zero-crossings around the mean."""
    mean_val = np.mean(signal)
    zero_crossings = np.where(np.diff(np.sign(signal - mean_val)))[0]
    return len(zero_crossings) / max(1, len(signal))


def approx_entropy(U, m=2, r=0.2):
    """Approximate entropy (simple implementation)."""
    N = len(U)
    if N <= m + 1:
        return 0.0

    def _phi(m_):
        x = np.array([U[i : i + m_] for i in range(N - m_ + 1)])
        C = (
                np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0)
                / (N - m_ + 1.0)
        )
        return np.sum(np.log(C + 1e-8)) / (N - m_ + 1.0)

    return abs(_phi(m) - _phi(m + 1))


def renyi_entropy(signal, alpha=2):
    """Rényi entropy on an empirical histogram."""
    hist, _ = np.histogram(signal, bins=64, density=True)
    probs = hist[hist > 0]
    return 1 / (1 - alpha) * np.log(np.sum(probs**alpha) + 1e-8)


def extract_bandpower(data, sf, band):
    """Mean Welch power in [fmin, fmax] per channel."""
    fmin, fmax = band
    psds = []
    for ch in data:
        f, Pxx = welch(ch, sf, nperseg=256)
        m = (f >= fmin) & (f <= fmax)
        psds.append(np.mean(Pxx[m]))
    return np.array(psds)


def sample_entropy(signal):
    """Cheap-ish proxy for entropy/irregularity (kept as-is from original)."""
    try:
        diff = np.abs(np.diff(signal))
        return np.log(np.std(diff) + 1e-8)
    except Exception:
        return 0.0


def higuchi_fd(signal, kmax=5):
    """Higuchi fractal dimension (lightweight version)."""
    L = []
    x = signal
    N = len(x)
    for k in range(1, kmax + 1):
        Lk = 0
        for m in range(k):
            denom = max(1, int(np.floor((N - m) / k))) * k
            Lmk = np.sum(np.abs(np.diff(x[m::k]))) * (N - 1) / max(1, denom)
            Lk += Lmk
        L.append(np.log((Lk / max(1, k)) + 1e-12))
    return float(np.mean(L))


def extract_epoch_features(epoch_data, sfreq, ch_names):
    """
    Per-epoch feature vector.
    Ordering must match make_epoch_feature_names().
    """
    features = []

    # A) Relative bandpowers
    total_power = extract_bandpower(epoch_data, sfreq, (0.5, 45))
    for band in [(4, 8), (8, 12), (12, 30), (0.5, 4), (30, 45)]:
        power = extract_bandpower(epoch_data, sfreq, band)
        features.extend(power / (total_power + 1e-8))

    # B) Stats + nonlinearities per channel
    for ch in epoch_data:
        features.append(np.mean(ch))
        features.append(np.std(ch))
        features.append(skew(ch))
        features.append(kurtosis(ch))
        features.append(sample_entropy(ch))
        features.append(higuchi_fd(ch))
        features.append(c0_complexity(ch))
        features.append(
            shannon_entropy(np.histogram(ch, bins=64, density=True)[0] + 1e-8)
        )
        features.append(approx_entropy(ch))
        features.append(renyi_entropy(ch))

    # C) Ratios (mean over channels)
    theta = extract_bandpower(epoch_data, sfreq, (4, 8))
    alpha = extract_bandpower(epoch_data, sfreq, (8, 12))
    beta = extract_bandpower(epoch_data, sfreq, (12, 30))
    features.append(np.mean(theta / (alpha + 1e-6)))
    features.append(np.mean(theta / (beta + 1e-6)))

    # D) Coherence features (alpha band)
    features.extend(compute_coherence_matrix(epoch_data, sfreq, band=(8, 12)))

    # E) Asymmetry (F3–F4 alpha)
    try:
        f3_idx = ch_names.index("F3")
        f4_idx = ch_names.index("F4")
        alpha_p = extract_bandpower(epoch_data, sfreq, (8, 12))
        features.append(
            (alpha_p[f4_idx] - alpha_p[f3_idx])
            / (alpha_p[f4_idx] + alpha_p[f3_idx] + 1e-6)
        )
    except ValueError:
        features.append(0.0)

    return features
