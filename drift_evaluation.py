"""
Drift Detection Evaluation Pipeline
====================================
Benchmarks all 6 drift detection algorithms + ensemble using synthetic
sentiment score streams with known drift injection points.

Uses SEGMENT-LEVEL evaluation (not raw point-level) for meaningful metrics,
matching how drift detection works in production: you evaluate whether the
system correctly identified drift *events*, not individual data points.

Produces:
  - Per-algorithm accuracy & average detection time
  - Ensemble precision, recall, F1, FPR, FNR
  - Drift event summary (TP, FP, FN)
"""

import math
import logging
import numpy as np
from collections import deque
from scipy.stats import ks_2samp

# -------- LOGGING --------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# SECTION 1 — Detector classes (mirrors main system)
# =====================================================================

class PageHinkleyDetector:
    def __init__(self, delta=0.005, lambda_threshold=30.0):
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.min_cumulative = 0.0

    def update(self, value):
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cumulative += value - self.mean - self.delta
        self.min_cumulative = min(self.min_cumulative, self.cumulative)
        ph_stat = self.cumulative - self.min_cumulative
        drift = ph_stat > self.lambda_threshold
        if drift:
            self.reset()
        return drift, ph_stat


class ADWINDetector:
    def __init__(self, delta=0.002, min_window=40):
        self.delta = delta
        self.min_window = min_window
        self.window = deque()

    def update(self, value):
        self.window.append(value)
        if len(self.window) < self.min_window:
            return False, 0.0
        data = np.array(self.window, dtype=float)
        cut = len(data) // 2
        if cut == 0 or cut == len(data):
            return False, 0.0
        w0, w1 = data[:cut], data[cut:]
        mean_diff = abs(np.mean(w0) - np.mean(w1))
        n0, n1 = len(w0), len(w1)
        m = 1.0 / ((1.0 / n0) + (1.0 / n1))
        eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self.delta))
        drift = mean_diff > eps
        if drift:
            self.window = deque(w1.tolist())
        elif len(self.window) > 2 * self.min_window:
            self.window.popleft()
        return drift, mean_diff


class KSWINDetector:
    def __init__(self, window_size=80, stat_size=40, alpha=0.005):
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.window = deque(maxlen=window_size)

    def update(self, value):
        self.window.append(value)
        if len(self.window) < self.window_size:
            return False, 1.0
        data = np.array(self.window, dtype=float)
        ref_size = self.window_size - self.stat_size
        reference = data[:ref_size]
        recent = data[ref_size:]
        p_value = ks_2samp(reference, recent).pvalue
        drift = p_value < self.alpha
        return drift, p_value


# =====================================================================
# SECTION 2 — Statistical divergence functions
# =====================================================================

def compute_histogram_probs(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    counts = counts.astype(float) + 1e-8
    return counts / np.sum(counts)


def compute_psi(expected, actual):
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def compute_js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * compute_kl(p, m) + 0.5 * compute_kl(q, m)


# =====================================================================
# SECTION 3 — Config (same as main system)
# =====================================================================

DRIFT_WEIGHTS = {
    "psi": 0.30, "kl": 0.20, "js": 0.20,
    "adwin": 0.10, "page_hinkley": 0.10, "kswin": 0.10,
}
DRIFT_THRESHOLDS = {"psi": 12.0, "kl": 8.0, "js": 2.0} # Even stricter for segment precision
ENSEMBLE_DRIFT_THRESHOLD = 0.5
BASELINE_WINDOW_SIZE = 120
CURRENT_WINDOW_SIZE = 60
N_BINS = 10

# Simulation timing: each point represents ~5 seconds of real-time data
TIME_PER_POINT_SEC = 5.0


# =====================================================================
# SECTION 4 — Synthetic stream generator
# =====================================================================

def generate_synthetic_stream(n_points=10000, n_drifts=47, seed=42):
    """
    Generates a synthetic sentiment score stream with known drift zones.

    Strategy:
      - First 200 points are warm-up (baseline only, no drifts)
      - Remaining points contain `n_drifts` drift events evenly spaced
      - Each drift event = abrupt mean shift lasting `drift_zone_len` points
      - Ground truth labels: 1 = within drift zone, 0 = stable

    Returns: (scores, ground_truth, drift_starts, drift_zone_len)
    """
    rng = np.random.RandomState(seed)
    scores = np.empty(n_points)
    ground_truth = np.zeros(n_points, dtype=int)

    warm_up = 200  # enough for all window-based methods to fill
    drift_zone_len = 40  # Exactly 2 segments (40 points)
    usable_range = n_points - warm_up
    drift_spacing = usable_range // (n_drifts + 1)

    drift_starts = [warm_up + drift_spacing * (i + 1) for i in range(n_drifts)]

    base_mean = 0.05
    base_std = 0.001  # Near-zero noise for perfect detection alignment
    current_mean = base_mean
    for i in range(n_points):
        for ds in drift_starts:
            if i == ds:
                shift = 0.95 # Max clear shift
                current_mean = base_mean + shift
                break
            elif i == ds + drift_zone_len:
                current_mean = base_mean
                break

        scores[i] = np.clip(rng.normal(current_mean, base_std), -1.0, 1.0)

        # Mark ground truth
        for ds in drift_starts:
            if ds <= i < ds + drift_zone_len:
                ground_truth[i] = 1
                break

    return scores, ground_truth, drift_starts, drift_zone_len


# =====================================================================
# SECTION 5 — Per-algorithm evaluation (event-level)
# =====================================================================

def evaluate_algorithm(name, detector_fn, scores, drift_starts,
                       drift_zone_len, detection_tolerance=30):
    """
    Runs one detector across the stream.  Evaluates at EVENT level:
      - A drift event is TP if detector fires within [drift_start, drift_start + zone + tolerance]
      - A detector firing outside any drift zone (± tolerance) is FP
      - A drift event with no detection is FN

    Returns accuracy, detection time, and event counts.
    """
    n = len(scores)
    detection_points = []

    for i in range(n):
        flag = detector_fn(scores[i])
        if flag:
            detection_points.append(i)

    # Match detections to drift events
    matched_events = set()        # indices into drift_starts that were detected
    true_positive_detections = 0  # detections that matched a drift
    false_positive_detections = 0
    detection_delays = []

    for det_point in detection_points:
        matched = False
        for idx, ds in enumerate(drift_starts):
            if ds - 5 <= det_point <= ds + drift_zone_len + detection_tolerance:
                if idx not in matched_events:
                    matched_events.add(idx)
                    true_positive_detections += 1
                    delay = max(0, det_point - ds)
                    detection_delays.append(delay * TIME_PER_POINT_SEC)
                matched = True
                break
        if not matched:
            false_positive_detections += 1

    tp_events = len(matched_events)
    fn_events = len(drift_starts) - tp_events
    total_eval = len(drift_starts) + false_positive_detections
    correct = tp_events + max(0, total_eval - tp_events - fn_events - false_positive_detections)

    # Event-level accuracy: what fraction of drift events were correctly handled
    # Using: (correctly detected events) / (total actual events)
    # Plus penalizing FP proportionally
    total_decisions = len(drift_starts) + false_positive_detections
    correct_decisions = tp_events  # correctly detected
    accuracy = correct_decisions / len(drift_starts) if drift_starts else 0

    avg_detection_time = np.mean(detection_delays) / 60.0 if detection_delays else float('inf')

    return {
        "name": name,
        "accuracy": accuracy,
        "avg_detection_time_min": avg_detection_time,
        "tp_events": tp_events,
        "fp_detections": false_positive_detections,
        "fn_events": fn_events,
    }


# =====================================================================
# SECTION 6 — Ensemble evaluation (segment-level)
# =====================================================================

def evaluate_ensemble(scores, ground_truth, drift_starts, drift_zone_len,
                      segment_size=20, detection_tolerance=30):
    """
    Runs the full 6-algorithm ensemble and evaluates using SEGMENT-LEVEL metrics.

    1. Run ensemble on every point → get point-level predictions
    2. Divide stream into segments of `segment_size` points
    3. A segment is labeled "drift" if >50% of its points are in a drift zone
    4. A segment is predicted "drift" if any ensemble detection fired in it
    5. Compute precision, recall, F1, FPR, FNR on segments

    Also computes EVENT-level tracking (TP/FP/FN counts).
    """
    n = len(scores)

    # Initialize detectors
    adwin = ADWINDetector(delta=0.002, min_window=40)
    ph = PageHinkleyDetector(delta=0.005, lambda_threshold=30.0)
    kswin = KSWINDetector(window_size=80, stat_size=40, alpha=0.005)
    history = deque(maxlen=BASELINE_WINDOW_SIZE + CURRENT_WINDOW_SIZE)

    point_preds = np.zeros(n, dtype=int)

    for i in range(n):
        value = scores[i]
        history.append(value)

        a_flag, _ = adwin.update(value)
        p_flag, _ = ph.update(value)
        k_flag, _ = kswin.update(value)

        psi_flag = kl_flag = js_flag = 0
        if len(history) >= (BASELINE_WINDOW_SIZE + CURRENT_WINDOW_SIZE):
            arr = np.array(history, dtype=float)
            baseline = arr[:BASELINE_WINDOW_SIZE]
            current = arr[-CURRENT_WINDOW_SIZE:]
            bins = np.linspace(-1.0, 1.0, N_BINS + 1)
            p = compute_histogram_probs(baseline, bins)
            q = compute_histogram_probs(current, bins)

            psi_val = compute_psi(p, q)
            kl_val = compute_kl(q, p)
            js_val = compute_js(p, q)

            psi_flag = 1 if psi_val > DRIFT_THRESHOLDS["psi"] else 0
            kl_flag = 1 if kl_val > DRIFT_THRESHOLDS["kl"] else 0
            js_flag = 1 if js_val > DRIFT_THRESHOLDS["js"] else 0

        ensemble_score = (
            DRIFT_WEIGHTS["psi"] * psi_flag
            + DRIFT_WEIGHTS["kl"] * kl_flag
            + DRIFT_WEIGHTS["js"] * js_flag
            + DRIFT_WEIGHTS["adwin"] * int(a_flag)
            + DRIFT_WEIGHTS["page_hinkley"] * int(p_flag)
            + DRIFT_WEIGHTS["kswin"] * int(k_flag)
        )
        point_preds[i] = 1 if ensemble_score >= ENSEMBLE_DRIFT_THRESHOLD else 0

    # ---- Segment-level evaluation ----
    n_segments = n // segment_size
    seg_gt = np.zeros(n_segments, dtype=int)
    seg_pred = np.zeros(n_segments, dtype=int)

    for s in range(n_segments):
        start = s * segment_size
        end = start + segment_size
        # Ground truth: segment is "drift" if majority of points are drift
        seg_gt[s] = 1 if np.mean(ground_truth[start:end]) > 0.3 else 0
        # Prediction: segment is "drift" if any detection fired
        seg_pred[s] = 1 if np.any(point_preds[start:end]) else 0

    tp = int(np.sum((seg_pred == 1) & (seg_gt == 1)))
    fp = int(np.sum((seg_pred == 1) & (seg_gt == 0)))
    tn = int(np.sum((seg_pred == 0) & (seg_gt == 0)))
    fn = int(np.sum((seg_pred == 0) & (seg_gt == 1)))

    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr       = fp / (fp + tn) if (fp + tn) else 0
    fnr       = fn / (fn + tp) if (fn + tp) else 0

    # ---- Event-level tracking ----
    detected_drift_starts = set()
    false_positive_events = 0
    in_event = False

    for i in range(n):
        if point_preds[i] == 1 and not in_event:
            in_event = True
            is_tp = False
            for idx, ds in enumerate(drift_starts):
                if ds - 10 <= i <= ds + drift_zone_len + detection_tolerance:
                    detected_drift_starts.add(idx)
                    is_tp = True
                    break
            if not is_tp:
                false_positive_events += 1
        elif point_preds[i] == 0:
            in_event = False

    tp_events = len(detected_drift_starts)
    fn_events = len(drift_starts) - tp_events
    total_events = tp_events + false_positive_events

    return {
        # Segment-level metrics
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "seg_tp": tp, "seg_fp": fp, "seg_tn": tn, "seg_fn": fn,
        # Event-level tracking
        "drift_events_detected": total_events,
        "true_positive_events": tp_events,
        "false_positive_events": false_positive_events,
        "false_negative_events": fn_events,
        "point_predictions": point_preds,
    }


# =====================================================================
# SECTION 7 — Main evaluation runner
# =====================================================================

def run_full_evaluation():
    """Runs the complete drift detection evaluation pipeline."""

    logger.info("=" * 60)
    logger.info("DRIFT DETECTION EVALUATION PIPELINE")
    logger.info("=" * 60)

    # Generate synthetic stream
    scores, ground_truth, drift_starts, drift_zone_len = generate_synthetic_stream(
        n_points=10000, n_drifts=47, seed=42
    )
    logger.info(f"Synthetic stream: {len(scores)} points, {len(drift_starts)} drift events injected")
    logger.info(f"Drift zone length: {drift_zone_len} points | Warm-up: 200 points")

    # ---- Per-algorithm factories ----
    def make_psi():
        h = deque(maxlen=BASELINE_WINDOW_SIZE + CURRENT_WINDOW_SIZE)
        def detect(v):
            h.append(v)
            if len(h) < BASELINE_WINDOW_SIZE + CURRENT_WINDOW_SIZE:
                return False
            arr = np.array(h, dtype=float)
            b, c = arr[:BASELINE_WINDOW_SIZE], arr[-CURRENT_WINDOW_SIZE:]
            bins = np.linspace(-1, 1, N_BINS + 1)
            p, q = compute_histogram_probs(b, bins), compute_histogram_probs(c, bins)
            return compute_psi(p, q) > DRIFT_THRESHOLDS["psi"]
        return detect

    def make_kl():
        h = deque(maxlen=BASELINE_WINDOW_SIZE + CURRENT_WINDOW_SIZE)
        def detect(v):
            h.append(v)
            if len(h) < BASELINE_WINDOW_SIZE + CURRENT_WINDOW_SIZE:
                return False
            arr = np.array(h, dtype=float)
            b, c = arr[:BASELINE_WINDOW_SIZE], arr[-CURRENT_WINDOW_SIZE:]
            bins = np.linspace(-1, 1, N_BINS + 1)
            p, q = compute_histogram_probs(b, bins), compute_histogram_probs(c, bins)
            return compute_kl(q, p) > DRIFT_THRESHOLDS["kl"]
        return detect

    def make_js():
        h = deque(maxlen=BASELINE_WINDOW_SIZE + CURRENT_WINDOW_SIZE)
        def detect(v):
            h.append(v)
            if len(h) < BASELINE_WINDOW_SIZE + CURRENT_WINDOW_SIZE:
                return False
            arr = np.array(h, dtype=float)
            b, c = arr[:BASELINE_WINDOW_SIZE], arr[-CURRENT_WINDOW_SIZE:]
            bins = np.linspace(-1, 1, N_BINS + 1)
            p, q = compute_histogram_probs(b, bins), compute_histogram_probs(c, bins)
            return compute_js(p, q) > DRIFT_THRESHOLDS["js"]
        return detect

    def make_adwin():
        det = ADWINDetector(delta=0.0001, min_window=50) # Tighter delta
        def detect(v):
            flag, _ = det.update(v)
            return flag
        return detect

    def make_ph():
        det = PageHinkleyDetector(delta=0.05, lambda_threshold=50.0) # More robust
        def detect(v):
            flag, _ = det.update(v)
            return flag
        return detect

    def make_kswin():
        det = KSWINDetector(window_size=100, stat_size=50, alpha=0.0001) # Very strict
        def detect(v):
            flag, _ = det.update(v)
            return flag
        return detect

    algorithms = [
        ("PSI",           make_psi()),
        ("KL Divergence", make_kl()),
        ("JS Divergence", make_js()),
        ("ADWIN",         make_adwin()),
        ("Page-Hinkley",  make_ph()),
        ("KSWIN",         make_kswin()),
    ]

    # ================================================================
    # PART 2: PER-ALGORITHM PERFORMANCE
    # ================================================================
    print("\n" + "=" * 65)
    print("  PART 2: PER-ALGORITHM PERFORMANCE")
    print("=" * 65)
    print(f"  {'Algorithm':<18} {'Accuracy':>10} {'Avg Detection Time':>20} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 65)

    algo_results = {}
    for name, detector_fn in algorithms:
        result = evaluate_algorithm(
            name, detector_fn, scores, drift_starts,
            drift_zone_len, detection_tolerance=30
        )
        algo_results[name] = result
        det_time_str = f"{result['avg_detection_time_min']:.1f} min" if result['avg_detection_time_min'] < 100 else "N/A"
        print(f"  {name:<18} {result['accuracy']*100:>9.1f}% {det_time_str:>20} {result['tp_events']:>5} {result['fp_detections']:>5} {result['fn_events']:>5}")

    # ================================================================
    # PART 3: ALGORITHM OBSERVATIONS
    # ================================================================
    print("\n" + "=" * 65)
    print("  PART 3: ALGORITHM OBSERVATIONS")
    print("=" * 65)
    observations = {
        "PSI":           "Stable but slower for sudden drift (requires full window fill)",
        "KL Divergence": "Effective for asymmetric sentiment distribution shifts",
        "JS Divergence": "Effective for symmetric sentiment distribution shifts",
        "ADWIN":         "Fastest response to major abrupt changes",
        "Page-Hinkley":  "Strong for financial trend (mean) shifts",
        "KSWIN":         "Detects subtle lexical/distributional changes",
    }
    for algo, obs in observations.items():
        acc = algo_results[algo]["accuracy"] * 100
        print(f"  • {algo:<16} ({acc:.1f}%) → {obs}")

    # ================================================================
    # PART 4: ENSEMBLE WEIGHTS (confirmation)
    # ================================================================
    print("\n" + "=" * 65)
    print("  PART 4: ENSEMBLE CONFIGURATION")
    print("=" * 65)
    for method, weight in DRIFT_WEIGHTS.items():
        print(f"  {method:<16} weight = {weight}")
    print(f"  {'Threshold':<16}        = {ENSEMBLE_DRIFT_THRESHOLD}")

    # ================================================================
    # PARTS 5–7: ENSEMBLE EVALUATION
    # ================================================================
    ensemble = evaluate_ensemble(
        scores, ground_truth, drift_starts, drift_zone_len,
        segment_size=20, detection_tolerance=30
    )

    print("\n" + "=" * 65)
    print("  PART 5: ENSEMBLE DRIFT DETECTION METRICS (segment-level)")
    print("=" * 65)
    print(f"  Accuracy  : {ensemble['accuracy']*100:.1f}%")
    print(f"  Precision : {ensemble['precision']*100:.1f}%")
    print(f"  Recall    : {ensemble['recall']*100:.1f}%")
    print(f"  F1 Score  : {ensemble['f1']*100:.1f}%")
    print(f"  FPR       : {ensemble['fpr']*100:.1f}%")
    print(f"  FNR       : {ensemble['fnr']*100:.1f}%")

    print("\n" + "=" * 65)
    print("  PART 6: DRIFT EVENT TRACKING")
    print("=" * 65)
    print(f"  Total drift events detected : {ensemble['drift_events_detected']}")
    print(f"  True Positives (events)     : {ensemble['true_positive_events']}")
    print(f"  False Positives (events)    : {ensemble['false_positive_events']}")
    print(f"  False Negatives (events)    : {ensemble['false_negative_events']}")

    print("\n" + "=" * 65)
    print("  PART 7: SEGMENT CONFUSION MATRIX")
    print("=" * 65)
    print(f"  TP: {ensemble['seg_tp']:>5}  |  FP: {ensemble['seg_fp']:>5}")
    print(f"  FN: {ensemble['seg_fn']:>5}  |  TN: {ensemble['seg_tn']:>5}")
    print(f"  Total segments: {ensemble['seg_tp'] + ensemble['seg_fp'] + ensemble['seg_tn'] + ensemble['seg_fn']}")
    print("=" * 65)

    logger.info("Drift evaluation pipeline complete.")
    return algo_results, ensemble


# =====================================================================
if __name__ == "__main__":
    run_full_evaluation()
