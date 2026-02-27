import ast
import math
import os
import pickle
import re

import numpy as np
import torch

# Lazy import for SASRec - only needed for sasrec reward type
SASRec = None

def _get_sasrec():
    global SASRec
    if SASRec is None:
        from sasrec import SASRec as _SASRec
        SASRec = _SASRec
    return SASRec


_SID_INFO_CACHE = None
_TITLE_TO_SID_CACHE = None
_EMBED_CACHE = None
_SASREC_CACHE = None


def _normalize_sid(value):
    if value is None:
        return ""
    return str(value).strip().strip("\n").strip("\"").strip("'")


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip().strip("\n").strip("\"").strip("'").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _strip_quotes(value):
    if value is None:
        return ""
    return str(value).strip().strip("\n").strip("\"").strip("'")


def _map_option_to_candidate(solution_str, extra_info):
    if not extra_info:
        return solution_str
    option_letters = extra_info.get("option_letters")
    candidates = extra_info.get("candidates")
    if not option_letters or not candidates or len(option_letters) != len(candidates):
        return solution_str
    solution_upper = str(solution_str or "").upper()
    # Handle numeric and letter options
    numeric_tokens = re.findall(r"\d+", solution_upper)
    if numeric_tokens:
        for token in reversed(numeric_tokens):
            if token in option_letters:
                idx = option_letters.index(token)
                if 0 <= idx < len(candidates):
                    return candidates[idx]
    letter_tokens = re.findall(r"[A-Z]+", solution_upper)
    for token in reversed(letter_tokens):
        if token in option_letters:
            idx = option_letters.index(token)
            if 0 <= idx < len(candidates):
                return candidates[idx]
    return solution_str


def _load_sid_info():
    global _SID_INFO_CACHE
    if _SID_INFO_CACHE is not None:
        return _SID_INFO_CACHE

    info_path = os.environ.get("SID_INFO_FILE", "")
    if not info_path:
        _SID_INFO_CACHE = ([], {})
        return _SID_INFO_CACHE

    with open(info_path, "r", encoding="utf-8") as f:
        info = f.readlines()
    item_name = [line.split("\t")[0].strip() for line in info]
    item2id = {name: i for i, name in enumerate(item_name)}
    _SID_INFO_CACHE = (item_name, item2id)
    return _SID_INFO_CACHE


def _load_title_to_sid():
    global _TITLE_TO_SID_CACHE
    if _TITLE_TO_SID_CACHE is not None:
        return _TITLE_TO_SID_CACHE

    info_path = os.environ.get("SID_INFO_FILE", "")
    if not info_path:
        _TITLE_TO_SID_CACHE = {}
        return _TITLE_TO_SID_CACHE

    title_to_sid = {}
    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sid = parts[0].strip()
            title = _normalize_text(parts[1])
            if title and sid:
                title_to_sid[title] = sid
    _TITLE_TO_SID_CACHE = title_to_sid
    return _TITLE_TO_SID_CACHE


def _map_text_to_sid(value):
    sid = _normalize_sid(value)
    if sid:
        return sid
    title = _normalize_text(value)
    if not title:
        return ""
    return _load_title_to_sid().get(title, "")


def _load_embeddings():
    global _EMBED_CACHE
    if _EMBED_CACHE is not None:
        return _EMBED_CACHE

    ada_path = os.environ.get("ADA_PATH", "")
    if not ada_path:
        return None
    with open(ada_path, "rb") as f:
        emb = pickle.load(f)
    emb = torch.tensor(emb, dtype=torch.float32)
    _EMBED_CACHE = emb
    return _EMBED_CACHE


def _load_sasrec():
    global _SASREC_CACHE
    if _SASREC_CACHE is not None:
        return _SASREC_CACHE

    cf_path = os.environ.get("SASREC_PATH", "")
    if not cf_path:
        return None

    item_name, _ = _load_sid_info()
    if not item_name:
        return None

    item_num = len(item_name)
    len_seq = int(os.environ.get("SASREC_LEN_SEQ", "10"))
    SASRecClass = _get_sasrec()
    model = SASRecClass(32, item_num, len_seq, 0.3, torch.device("cpu"))
    model.load_state_dict(torch.load(cf_path, map_location="cpu"))
    model.eval()
    _SASREC_CACHE = model
    return _SASREC_CACHE


def _get_history(extra_info):
    if not extra_info:
        return []
    history = extra_info.get("history_item_sid", [])
    if isinstance(history, str):
        try:
            history = ast.literal_eval(history)
        except Exception:
            history = []
    return history


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return compute_score_rule(data_source, solution_str, ground_truth, extra_info)


def compute_score_rule(data_source, solution_str, ground_truth, extra_info=None):
    pred = _map_text_to_sid(solution_str)
    target = _map_text_to_sid(ground_truth)
    return 1.0 if pred == target else 0.0


def compute_score_ndcg(data_source, solution_str, ground_truth, extra_info=None):
    pred = _map_text_to_sid(solution_str)
    target = _map_text_to_sid(ground_truth)
    if pred == target:
        return 1.0
    rank = None if not extra_info else extra_info.get("rank", None)
    if rank is None:
        return 0.0
    try:
        rank = int(rank)
    except Exception:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def compute_score_ranking(data_source, solution_str, ground_truth, extra_info=None):
    return compute_score_rule(data_source, solution_str, ground_truth, extra_info) + compute_score_ndcg(
        data_source, solution_str, ground_truth, extra_info
    )


def compute_score_ranking_only(data_source, solution_str, ground_truth, extra_info=None):
    return compute_score_ndcg(data_source, solution_str, ground_truth, extra_info)


def compute_score_semantic(data_source, solution_str, ground_truth, extra_info=None):
    item_name, item2id = _load_sid_info()
    if not item_name:
        return 0.0
    emb = _load_embeddings()
    if emb is None:
        return 0.0

    pred = _map_text_to_sid(solution_str)
    target = _map_text_to_sid(ground_truth)
    if pred not in item2id or target not in item2id:
        return 0.0

    pred_vec = emb[item2id[pred]]
    target_vec = emb[item2id[target]]
    pred_vec = pred_vec / (pred_vec.norm() + 1e-8)
    target_vec = target_vec / (target_vec.norm() + 1e-8)
    return float(torch.sum(pred_vec * target_vec).item())


def compute_score_sasrec(data_source, solution_str, ground_truth, extra_info=None):
    model = _load_sasrec()
    if model is None:
        return 0.0

    item_name, item2id = _load_sid_info()
    if not item_name:
        return 0.0

    pred = _map_text_to_sid(solution_str)
    if pred not in item2id:
        return 0.0

    history = _get_history(extra_info)
    history_ids = []
    for sid in history:
        sid = _normalize_sid(sid)
        if sid in item2id:
            history_ids.append(item2id[sid])

    len_seq = int(os.environ.get("SASREC_LEN_SEQ", "10"))
    item_num = len(item_name)
    if len(history_ids) < len_seq:
        history_ids = history_ids + [item_num] * (len_seq - len(history_ids))

    seq = torch.LongTensor([history_ids])
    len_lis = torch.tensor(np.array([min(len(history), len_seq)]))
    pred_id = torch.LongTensor([item2id[pred]])

    with torch.no_grad():
        predictions = model.forward_eval(seq, len_lis)
        score = torch.gather(predictions, 1, pred_id.view(-1, 1)).view(-1)[0]
    return float(score.item())


def compute_score_mind_ndcg(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute nDCG-style reward for MIND news recommendation (text-based).

    This function supports both simple binary matching and full ranking evaluation.

    Args:
        data_source: Not used (kept for API compatibility)
        solution_str: LLM-generated news title (prediction)
        ground_truth: Ground truth clicked news title
        extra_info: Optional dict containing:
            - 'candidates': List[str] of all candidate news titles
            - 'labels': List[int] of binary labels (1=clicked, 0=not clicked)
            - 'rank': int - pre-computed rank (if available)

    Returns:
        float: Reward score in [0, 1] range
            - 1.0 if exact match with ground truth
            - 1/log2(rank+1) if found in top-10 candidates
            - 0.0 otherwise

    Examples:
        >>> # Exact match
        >>> compute_score_mind_ndcg(None, "Breaking News Story", "Breaking News Story", None)
        1.0

        >>> # Ranked at position 2
        >>> extra = {'rank': 2}
        >>> compute_score_mind_ndcg(None, "Some News", "Breaking News", extra)
        0.6309...  # 1/log2(3)
    """
    # Normalize text for comparison (case-insensitive, strip whitespace)
    solution_str = _map_option_to_candidate(solution_str, extra_info)
    solution_normalized = _strip_quotes(solution_str).lower() if solution_str else ""
    ground_truth_normalized = _strip_quotes(ground_truth).lower() if ground_truth else ""

    # Case 1: Exact match with ground truth (best case)
    if solution_normalized == ground_truth_normalized:
        return 1.0

    # Case 2: Use pre-computed rank if available
    if extra_info and 'rank' in extra_info:
        rank = extra_info.get('rank')
        if rank is None:
            return 0.0
        try:
            rank = int(rank)
            if rank <= 0:
                return 0.0
            # DCG-style reward: 1/log2(rank+1)
            # rank=1 -> 1.0, rank=2 -> 0.631, rank=3 -> 0.5, rank=5 -> 0.387, rank=10 -> 0.301
            if rank <= 10:
                return 1.0 / math.log2(rank + 1)
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0

    # Case 3: Compute rank from full candidate list
    if extra_info and 'candidates' in extra_info and 'labels' in extra_info:
        candidates = extra_info.get('candidates', [])
        labels = extra_info.get('labels', [])

        if not candidates or not labels or len(candidates) != len(labels):
            return 0.0

        # Find rank of the generated solution in candidate list
        rank = None
        for i, candidate in enumerate(candidates):
            candidate_normalized = candidate.strip().lower() if candidate else ""
            if candidate_normalized == solution_normalized:
                rank = i + 1  # 1-indexed rank
                break

        if rank is None:
            # Generated text doesn't match any candidate
            return 0.0

        # Check if this candidate was actually clicked
        if rank <= len(labels) and labels[rank - 1] == 1:
            # Clicked item - give DCG-style reward based on rank
            if rank <= 10:
                return 1.0 / math.log2(rank + 1)
            else:
                return 0.0
        else:
            # Not a clicked item - no reward
            return 0.0

    # Case 4: Fuzzy matching as fallback (if solution is substring of ground truth or vice versa)
    if solution_normalized and ground_truth_normalized:
        if solution_normalized in ground_truth_normalized or ground_truth_normalized in solution_normalized:
            # Partial match - give small reward (0.3)
            return 0.3

    # No match
    return 0.0


def compute_score_mind_mrr(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute MRR (Mean Reciprocal Rank) style reward for MIND.
    Similar to nDCG but uses 1/rank instead of 1/log2(rank+1).

    Args:
        Same as compute_score_mind_ndcg

    Returns:
        float: 1/rank if found in candidates, 0 otherwise
    """
    solution_str = _map_option_to_candidate(solution_str, extra_info)
    solution_normalized = _strip_quotes(solution_str).lower() if solution_str else ""
    ground_truth_normalized = _strip_quotes(ground_truth).lower() if ground_truth else ""

    if solution_normalized == ground_truth_normalized:
        return 1.0

    if extra_info and 'rank' in extra_info:
        rank = extra_info.get('rank')
        try:
            rank = int(rank)
            if rank > 0:
                return 1.0 / rank
        except (ValueError, TypeError):
            pass

    if extra_info and 'candidates' in extra_info:
        candidates = extra_info.get('candidates', [])
        for i, candidate in enumerate(candidates):
            candidate_normalized = candidate.strip().lower() if candidate else ""
            if candidate_normalized == solution_normalized:
                return 1.0 / (i + 1)

    return 0.0


def compute_score_mind_auc(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute AUC-like reward for MIND news recommendation.

    This reward simulates AUC by computing the proportion of negative samples
    that would be ranked below the selected positive sample.

    The reward is designed to match the AUC evaluation metric used by MIND benchmark:
    - If model selects a clicked item, reward = (# non-clicked items) / total_non_clicked
      (i.e., fraction of negative items that would be ranked below)
    - If model selects a non-clicked item, reward = 0

    This encourages the model to rank clicked items above non-clicked items,
    which is exactly what AUC measures.

    Args:
        data_source: Not used (kept for API compatibility)
        solution_str: Model's predicted candidate index (as string number, e.g., "3")
        ground_truth: Ground truth clicked news title (for reference)
        extra_info: Dict containing:
            - 'candidates': List[str] of all candidate news titles
            - 'labels': List[int] of binary labels (1=clicked, 0=not clicked)

    Returns:
        float: AUC-like reward in [0, 1] range
            - 1.0 if selected item is clicked and all other items are non-clicked
            - Proportional reward based on position among negatives
            - 0.0 if selected item is not clicked or invalid

    Example:
        If there are 5 candidates with labels [0, 1, 0, 0, 1] and model selects
        candidate 1 (clicked), the reward would be 1.0 because selecting a clicked
        item is always correct.

        If we had a ranking-based version, selecting the higher-ranked clicked
        item would give higher reward, but for simplicity we treat all clicked
        items equally.
    """
    if not extra_info:
        return 0.0

    candidates = extra_info.get('candidates', [])
    labels = extra_info.get('labels', [])

    if not candidates or not labels or len(candidates) != len(labels):
        return 0.0

    # Parse the solution - expecting a number index
    solution_str = str(solution_str or "").strip()

    # First, try to map option letter to index (for backward compatibility)
    option_letters = extra_info.get('option_letters', [])
    selected_idx = None

    if option_letters:
        # Try letter-based selection (e.g., "A", "B", etc.)
        solution_upper = solution_str.upper()
        tokens = re.findall(r"[A-Z]+", solution_upper)
        for token in reversed(tokens):
            if token in option_letters:
                selected_idx = option_letters.index(token)
                break

    # Try numeric index if letter mapping failed
    if selected_idx is None:
        # Extract first number from solution
        numbers = re.findall(r"\d+", solution_str)
        if numbers:
            try:
                # Assume 1-indexed from model output
                selected_idx = int(numbers[0]) - 1
            except (ValueError, IndexError):
                pass

    # Validate index
    if selected_idx is None or selected_idx < 0 or selected_idx >= len(candidates):
        return 0.0

    # Check if selected candidate is clicked
    if labels[selected_idx] != 1:
        # Selected a non-clicked item - no reward
        return 0.0

    # Selected a clicked item - compute AUC-like reward
    # Count total positives and negatives
    num_positives = sum(labels)
    num_negatives = len(labels) - num_positives

    if num_negatives == 0:
        # All items are clicked (rare), perfect score
        return 1.0

    if num_positives == 0:
        # No clicked items (shouldn't happen with valid data)
        return 0.0

    # Basic AUC reward: selecting any clicked item gives positive reward
    # The reward is higher if there are more negatives to beat
    # This simulates: "what fraction of pairwise comparisons would we win?"

    # Simple version: binary reward for selecting clicked item
    # More sophisticated: could weight by position, but keep it simple for stability
    return 1.0


def compute_score_mind_auc_rank(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute AUC-like reward with ranking consideration for MIND.

    This is a more nuanced version that gives partial credit based on
    where the selected item ranks among clicked items.

    Args:
        Same as compute_score_mind_auc

    Returns:
        float: Reward in [0, 1] range
            - 1.0 if selected the first (most relevant) clicked item
            - Decreasing reward for lower-ranked clicked items
            - 0.0 if selected non-clicked item
    """
    if not extra_info:
        return 0.0

    candidates = extra_info.get('candidates', [])
    labels = extra_info.get('labels', [])

    if not candidates or not labels or len(candidates) != len(labels):
        return 0.0

    # Parse solution
    solution_str = str(solution_str or "").strip()
    option_letters = extra_info.get('option_letters', [])
    selected_idx = None

    if option_letters:
        solution_upper = solution_str.upper()
        tokens = re.findall(r"[A-Z]+", solution_upper)
        for token in reversed(tokens):
            if token in option_letters:
                selected_idx = option_letters.index(token)
                break

    if selected_idx is None:
        numbers = re.findall(r"\d+", solution_str)
        if numbers:
            try:
                selected_idx = int(numbers[0]) - 1
            except (ValueError, IndexError):
                pass

    if selected_idx is None or selected_idx < 0 or selected_idx >= len(candidates):
        return 0.0

    if labels[selected_idx] != 1:
        return 0.0

    # Find rank of selected item among clicked items
    clicked_indices = [i for i, l in enumerate(labels) if l == 1]
    if not clicked_indices:
        return 0.0

    # Find position of selected index among clicked items
    click_rank = clicked_indices.index(selected_idx) + 1  # 1-indexed
    num_clicked = len(clicked_indices)

    # Higher reward for selecting earlier clicked items
    # rank 1 -> 1.0, rank 2 -> 0.75, rank 3 -> 0.67, etc.
    # Using 1/log2(rank+1) style decay
    reward = 1.0 / math.log2(click_rank + 1)

    return reward


# =============================================================================
# POINTWISE REWARD FUNCTIONS
# =============================================================================
# These rewards are for pointwise RL where model outputs "Yes" or "No"
# for each (history, candidate) pair.


def compute_score_pointwise_binary(data_source, solution_str, ground_truth, extra_info=None):
    """
    Simple binary reward for pointwise Yes/No prediction.

    Args:
        data_source: Not used (kept for API compatibility)
        solution_str: Model's prediction ("Yes" or "No")
        ground_truth: Expected answer ("Yes" or "No")
        extra_info: Optional dict (not used for binary reward)

    Returns:
        float: 1.0 if prediction matches ground truth, 0.0 otherwise

    Example:
        >>> compute_score_pointwise_binary(None, "Yes", "Yes", None)
        1.0
        >>> compute_score_pointwise_binary(None, "No", "Yes", None)
        0.0
    """
    if not solution_str or not ground_truth:
        return 0.0

    # Normalize predictions
    pred_str = str(solution_str).strip().lower()
    target_str = str(ground_truth).strip().lower()

    # Check for Yes/No
    pred_yes = "yes" in pred_str
    target_yes = "yes" in target_str

    return 1.0 if pred_yes == target_yes else 0.0


def compute_score_pointwise_weighted(data_source, solution_str, ground_truth, extra_info=None):
    """
    Weighted reward for pointwise prediction that approximates AUC optimization.

    This reward gives higher weight to correctly predicting "Yes" for positive
    samples (clicked items) than correctly predicting "No" for negative samples.
    This aligns with AUC optimization where we care most about ranking positives
    above negatives.

    Args:
        data_source: Not used (kept for API compatibility)
        solution_str: Model's prediction ("Yes" or "No")
        ground_truth: Expected answer ("Yes" or "No")
        extra_info: Dict containing:
            - 'label': int (1 for clicked/positive, 0 for not clicked/negative)

    Returns:
        float: Weighted reward
            - Correct Yes on positive: 1.0
            - Correct No on negative: 0.5
            - Incorrect prediction: 0.0

    Rationale:
        - In AUC, we want P(Yes|positive) > P(Yes|negative)
        - Giving full reward (1.0) for correct positives encourages high P(Yes) for positives
        - Giving partial reward (0.5) for correct negatives encourages low P(Yes) for negatives
        - This asymmetry helps optimize ranking
    """
    if not solution_str or not ground_truth:
        return 0.0

    # Normalize prediction
    pred_str = str(solution_str).strip().lower()
    pred_yes = "yes" in pred_str

    # Get ground truth label from extra_info
    label = 0
    if extra_info:
        label = extra_info.get('label', 0)
    else:
        # Fallback: infer from ground_truth string
        target_str = str(ground_truth).strip().lower()
        label = 1 if "yes" in target_str else 0

    if label == 1:  # Positive sample (clicked item)
        return 1.0 if pred_yes else 0.0
    else:  # Negative sample (not clicked)
        return 0.5 if not pred_yes else 0.0


def compute_score_pointwise_auc_proxy(data_source, solution_str, ground_truth, extra_info=None):
    """
    AUC-proxy reward for pointwise prediction.

    This reward is designed to more directly optimize AUC by considering
    the confidence of the prediction. It rewards:
    - High confidence "Yes" for positives
    - High confidence "No" for negatives

    Args:
        data_source: Not used
        solution_str: Model's prediction ("Yes" or "No")
        ground_truth: Expected answer
        extra_info: Dict containing:
            - 'label': int (1=positive, 0=negative)
            - 'confidence': float (optional, 0-1 confidence score)

    Returns:
        float: Reward in [0, 1] range
    """
    if not solution_str:
        return 0.0

    pred_str = str(solution_str).strip().lower()
    pred_yes = "yes" in pred_str

    # Get label
    label = 0
    if extra_info:
        label = extra_info.get('label', 0)
    else:
        target_str = str(ground_truth).strip().lower()
        label = 1 if "yes" in target_str else 0

    # Get confidence if available (default to 1.0 for deterministic predictions)
    confidence = 1.0
    if extra_info:
        confidence = extra_info.get('confidence', 1.0)

    if label == 1:  # Positive sample
        if pred_yes:
            return confidence  # Reward proportional to confidence
        else:
            return 0.0  # Wrong prediction on positive = no reward
    else:  # Negative sample
        if not pred_yes:
            return 0.5 * confidence  # Partial reward for correct negative
        else:
            return 0.0  # Wrong prediction on negative = no reward


def compute_score_pointwise_asymmetric(data_source, solution_str, ground_truth, extra_info=None):
    """
    Asymmetric reward for pointwise prediction designed to prevent mode collapse.

    Key insight: With neg_ratio=3.0 (75% negatives, 25% positives):
      - Always "Yes" strategy: 0.25*1.0 + 0.75*(-1.0) = -0.50 (terrible)
      - Always "No" strategy:  0.25*(-0.3) + 0.75*0.5  = +0.30 (mediocre)
      - Perfect discrimination: 0.25*1.0 + 0.75*0.5    = +0.625 (optimal)

    The harsh false-positive penalty (-1.0) makes indiscriminate "Yes" predictions
    very costly, breaking the mode collapse that ruins pointwise_weighted.

    Args:
        data_source: Not used (kept for API compatibility)
        solution_str: Model's prediction ("Yes" or "No")
        ground_truth: Expected answer ("Yes" or "No")
        extra_info: Dict containing 'label' (1=positive, 0=negative)

    Returns:
        float: Reward in [-1.0, 1.0] range
            - Correct Yes on positive:  +1.0
            - Correct No on negative:   +0.5
            - Wrong Yes on negative:    -1.0 (harsh false positive penalty)
            - Wrong No on positive:     -0.3 (moderate false negative penalty)
    """
    if not solution_str:
        return -0.5

    pred_str = str(solution_str).strip().lower()
    pred_yes = "yes" in pred_str

    label = 0
    if extra_info:
        label = extra_info.get('label', 0)
    else:
        target_str = str(ground_truth).strip().lower()
        label = 1 if "yes" in target_str else 0

    if label == 1:  # Positive sample
        return 1.0 if pred_yes else -0.3
    else:  # Negative sample
        return -1.0 if pred_yes else 0.5


def compute_score_pointwise_margin(data_source, solution_str, ground_truth, extra_info=None):
    """
    Margin-based reward for pointwise prediction.

    This reward penalizes wrong predictions more than it rewards correct ones,
    creating a margin that encourages confident correct predictions.

    Args:
        Same as compute_score_pointwise_weighted

    Returns:
        float: Reward in [-0.5, 1.0] range
            - Correct Yes on positive: 1.0
            - Correct No on negative: 0.3
            - Wrong Yes on negative: -0.5 (penalty for false positive)
            - Wrong No on positive: -0.3 (penalty for false negative)
    """
    if not solution_str:
        return -0.5  # Penalty for no prediction

    pred_str = str(solution_str).strip().lower()
    pred_yes = "yes" in pred_str

    label = 0
    if extra_info:
        label = extra_info.get('label', 0)
    else:
        target_str = str(ground_truth).strip().lower()
        label = 1 if "yes" in target_str else 0

    if label == 1:  # Positive sample
        if pred_yes:
            return 1.0  # Correct: predicted Yes for clicked item
        else:
            return -0.3  # Wrong: missed a clicked item
    else:  # Negative sample
        if not pred_yes:
            return 0.3  # Correct: predicted No for non-clicked item
        else:
            return -0.5  # Wrong: false positive (predicted Yes for non-clicked)


# =============================================================================
# CHAIN-OF-THOUGHT (COT) REWARD FUNCTIONS FOR LIST-WISE RANKING
# =============================================================================
# New format: model outputs <think>reasoning</think><answer>[1:p, 2:p, ...]</answer>
# Old format (legacy): <reasoning> ... Answer: <number>
#
# The new format outputs click probabilities for ALL candidates, enabling:
# - AUC/nDCG computation over the full ranking
# - Richer reward signal (not just "right/wrong")
# - Format checking on structured output


def extract_cot_answer(solution_str, num_candidates=None):
    """
    Extract the final answer number from Chain-of-Thought output (LEGACY).

    Kept for backward compatibility with old-format CoT models.
    For the new <think>/<answer> format, use extract_cot_probs() instead.

    Returns:
        int or None: The extracted answer (1-indexed) or None if not found
    """
    if not solution_str:
        return None

    text = str(solution_str).strip()

    # If model used <think> tags, only look after </think>
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()

    # If new <answer> format with probs, return the argmax
    probs = extract_cot_probs(solution_str, num_candidates)
    if probs:
        best_id = max(probs, key=probs.get)
        return best_id

    # Pattern 1: "Answer: X"
    match = re.search(r'[Aa]nswer\s*:\s*(\d+)', text)
    if match:
        ans = int(match.group(1))
        if num_candidates is None or 1 <= ans <= num_candidates:
            return ans

    # Pattern 2: "The answer is X"
    match = re.search(r'[Tt]he\s+answer\s+is\s+(\d+)', text)
    if match:
        ans = int(match.group(1))
        if num_candidates is None or 1 <= ans <= num_candidates:
            return ans

    # Pattern 3: "I choose X" or "I select X" or "I pick X"
    match = re.search(r'[Ii]\s+(?:choose|select|pick)\s+(\d+)', text)
    if match:
        ans = int(match.group(1))
        if num_candidates is None or 1 <= ans <= num_candidates:
            return ans

    # Pattern 4: "Article X" or "Candidate X" or "Option X" at the end
    match = re.search(r'(?:[Aa]rticle|[Cc]andidate|[Oo]ption)\s+(\d+)\s*$', text)
    if match:
        ans = int(match.group(1))
        if num_candidates is None or 1 <= ans <= num_candidates:
            return ans

    # Pattern 5: Last number in the text (fallback)
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        ans = int(numbers[-1])
        if num_candidates is None or 1 <= ans <= num_candidates:
            return ans

    return None


def extract_cot_probs(solution_str, num_candidates=None):
    """
    Extract click probabilities from <answer>[1:0.8, 2:0.1, ...]</answer> format.

    Supports formats:
        <answer>[1:0.8, 2:0.1, 3:0.6]</answer>
        <answer>  [1: 0.8, 2: 0.1, 3: 0.6]  </answer>
        <answer>1:0.8, 2:0.1, 3:0.6</answer>   (without brackets)

    Args:
        solution_str: Full model output
        num_candidates: Optional max candidate ID for validation

    Returns:
        dict[int, float]: Mapping from candidate_id (1-indexed) to probability.
                          Empty dict if parsing fails.
    """
    if not solution_str:
        return {}

    text = str(solution_str).strip()

    # Extract content between <answer> and </answer>
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if not answer_match:
        return {}

    answer_content = answer_match.group(1).strip()

    # Remove optional brackets
    answer_content = answer_content.strip('[]')

    # Parse "id:prob" pairs
    # Supports: "1:0.8" "1: 0.8" "1 : 0.8" "1:0.8,"
    pairs = re.findall(r'(\d+)\s*:\s*([0-9]*\.?[0-9]+)', answer_content)
    if not pairs:
        return {}

    probs = {}
    for cid_str, prob_str in pairs:
        cid = int(cid_str)
        try:
            prob = float(prob_str)
        except ValueError:
            continue

        # Clamp probability to [0, 1]
        prob = max(0.0, min(1.0, prob))

        # Validate candidate ID
        if num_candidates is not None and (cid < 1 or cid > num_candidates):
            continue

        probs[cid] = prob

    return probs


def _check_cot_format(solution_str):
    """
    Check if the model output follows the <think>...</think><answer>...</answer> format.

    Returns:
        dict with keys:
            'has_think': bool - has <think> tags
            'has_answer': bool - has <answer> tags
            'has_reasoning': bool - reasoning content is non-trivial (>20 chars)
            'has_probs': bool - answer contains parseable probabilities
            'good_format': bool - all checks pass
    """
    if not solution_str:
        return {'has_think': False, 'has_answer': False, 'has_reasoning': False,
                'has_probs': False, 'good_format': False}

    text = str(solution_str).strip()

    has_think = bool(re.search(r'<think>.*?</think>', text, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', text, re.DOTALL))

    # Check reasoning quality
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    has_reasoning = bool(think_match and len(think_match.group(1).strip()) > 20)

    # Check if probs parse successfully
    probs = extract_cot_probs(text)
    has_probs = len(probs) > 0

    good_format = has_think and has_answer and has_reasoning and has_probs

    return {
        'has_think': has_think,
        'has_answer': has_answer,
        'has_reasoning': has_reasoning,
        'has_probs': has_probs,
        'good_format': good_format,
    }


# -----------------------------------------------------------------------------
# New reward functions (prob-based)
# -----------------------------------------------------------------------------

def compute_score_mind_cot_prob_auc(data_source, solution_str, ground_truth, extra_info=None):
    """
    AUC reward based on predicted click probabilities.

    Computes AUC: for each (clicked, non-clicked) pair, reward += 1 if
    predicted_prob(clicked) > predicted_prob(non-clicked).
    Normalized to [0, 1].

    Returns:
        float: AUC score in [0, 1], or 0.0 if format is invalid.
    """
    if not extra_info or 'labels' not in extra_info:
        return 0.0

    labels = extra_info.get('labels', [])
    num_candidates = extra_info.get('num_candidates', len(labels))

    probs = extract_cot_probs(solution_str, num_candidates)
    if not probs:
        return 0.0

    # Collect clicked and non-clicked probs
    clicked_probs = []
    non_clicked_probs = []
    for i, label in enumerate(labels):
        cid = i + 1  # 1-indexed
        p = probs.get(cid, 0.0)  # Default to 0 if candidate not mentioned
        if label == 1:
            clicked_probs.append(p)
        else:
            non_clicked_probs.append(p)

    if not clicked_probs or not non_clicked_probs:
        return 0.0

    # Compute AUC: fraction of (pos, neg) pairs correctly ordered
    correct = 0
    total = 0
    for cp in clicked_probs:
        for np_ in non_clicked_probs:
            total += 1
            if cp > np_:
                correct += 1
            elif cp == np_:
                correct += 0.5

    return correct / total if total > 0 else 0.0


def compute_score_mind_cot_prob_ndcg(data_source, solution_str, ground_truth, extra_info=None):
    """
    nDCG@k reward based on predicted click probabilities.

    Ranks candidates by predicted probability, computes nDCG using actual labels.

    Returns:
        float: nDCG score in [0, 1], or 0.0 if format is invalid.
    """
    import math

    if not extra_info or 'labels' not in extra_info:
        return 0.0

    labels = extra_info.get('labels', [])
    num_candidates = extra_info.get('num_candidates', len(labels))

    probs = extract_cot_probs(solution_str, num_candidates)
    if not probs:
        return 0.0

    # Build (candidate_id, predicted_prob, actual_label) list
    items = []
    for i, label in enumerate(labels):
        cid = i + 1
        p = probs.get(cid, 0.0)
        items.append((cid, p, label))

    # Sort by predicted prob descending
    items.sort(key=lambda x: x[1], reverse=True)

    # DCG
    dcg = 0.0
    for rank, (_, _, rel) in enumerate(items, 1):
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG (sort by actual label descending)
    ideal = sorted([l for _, _, l in items], reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(ideal, 1):
        idcg += rel / math.log2(rank + 1)

    return dcg / idcg if idcg > 0 else 0.0


def compute_score_mind_cot_prob_ce(data_source, solution_str, ground_truth, extra_info=None):
    """
    Cross-entropy reward based on predicted click probabilities.

    Reward = average of:
        - For clicked items: prob (higher is better)
        - For non-clicked items: 1 - prob (lower is better)
    Naturally in [0, 1].

    Returns:
        float: CE-based score in [0, 1], or 0.0 if format is invalid.
    """
    if not extra_info or 'labels' not in extra_info:
        return 0.0

    labels = extra_info.get('labels', [])
    num_candidates = extra_info.get('num_candidates', len(labels))

    probs = extract_cot_probs(solution_str, num_candidates)
    if not probs:
        return 0.0

    scores = []
    for i, label in enumerate(labels):
        cid = i + 1
        p = probs.get(cid, 0.0)
        # Clamp to avoid edge cases
        p = max(0.01, min(0.99, p))
        if label == 1:
            scores.append(p)         # want high prob for clicked
        else:
            scores.append(1.0 - p)   # want low prob for non-clicked

    return sum(scores) / len(scores) if scores else 0.0


def compute_score_mind_cot_prob_margin(data_source, solution_str, ground_truth, extra_info=None):
    """
    Margin reward: avg(clicked_prob) - avg(non_clicked_prob).

    Encourages separation between clicked and non-clicked probabilities.
    Mapped from [-1, 1] to [0, 1] via (margin + 1) / 2.

    Returns:
        float: Score in [0, 1], or 0.0 if format is invalid.
    """
    if not extra_info or 'labels' not in extra_info:
        return 0.0

    labels = extra_info.get('labels', [])
    num_candidates = extra_info.get('num_candidates', len(labels))

    probs = extract_cot_probs(solution_str, num_candidates)
    if not probs:
        return 0.0

    clicked_probs = []
    non_clicked_probs = []
    for i, label in enumerate(labels):
        cid = i + 1
        p = probs.get(cid, 0.0)
        if label == 1:
            clicked_probs.append(p)
        else:
            non_clicked_probs.append(p)

    if not clicked_probs or not non_clicked_probs:
        return 0.0

    avg_pos = sum(clicked_probs) / len(clicked_probs)
    avg_neg = sum(non_clicked_probs) / len(non_clicked_probs)
    margin = avg_pos - avg_neg  # in [-1, 1]

    return (margin + 1.0) / 2.0  # map to [0, 1]


def compute_score_mind_cot_prob_format(data_source, solution_str, ground_truth, extra_info=None):
    """
    Combined reward: AUC score + format bonus.

    - Good format (<think>+<answer> with valid probs): +0.1 bonus
    - Coverage bonus (listed all candidates): +0.05 bonus
    - Base: AUC reward

    Returns:
        float: Score in [0, 1.15] range (clamped to [0, 1]).
    """
    fmt = _check_cot_format(solution_str)

    # Base reward: AUC from probs
    base = compute_score_mind_cot_prob_auc(data_source, solution_str, ground_truth, extra_info)

    # Format bonus
    format_bonus = 0.1 if fmt['good_format'] else 0.0

    # Coverage bonus: did model list probabilities for all candidates?
    coverage_bonus = 0.0
    if extra_info and fmt['has_probs']:
        num_candidates = extra_info.get('num_candidates', 0)
        probs = extract_cot_probs(solution_str, num_candidates)
        if num_candidates > 0 and len(probs) >= num_candidates:
            coverage_bonus = 0.05

    return min(1.0, base + format_bonus + coverage_bonus)


# -----------------------------------------------------------------------------
# Legacy reward functions (single-answer based, kept for backward compatibility)
# -----------------------------------------------------------------------------

def compute_score_mind_cot_binary(data_source, solution_str, ground_truth, extra_info=None):
    """
    Binary reward for CoT output (legacy single-answer format).
    Also works with new prob format by taking argmax.
    """
    num_candidates = None
    if extra_info:
        num_candidates = extra_info.get('num_candidates')

    predicted = extract_cot_answer(solution_str, num_candidates)
    if predicted is None:
        return 0.0

    target = None
    if extra_info and 'clicked_idx' in extra_info:
        target = extra_info.get('clicked_idx')
    elif ground_truth:
        try:
            target = int(str(ground_truth).strip())
        except ValueError:
            match = re.search(r'(\d+)', str(ground_truth))
            if match:
                target = int(match.group(1))

    if target is None:
        return 0.0

    return 1.0 if predicted == target else 0.0


def compute_score_mind_cot_ndcg(data_source, solution_str, ground_truth, extra_info=None):
    """nDCG-style reward (legacy single-answer). Use cot_prob_ndcg for new format."""
    num_candidates = None
    if extra_info:
        num_candidates = extra_info.get('num_candidates')

    predicted = extract_cot_answer(solution_str, num_candidates)
    if predicted is None:
        return 0.0

    target = None
    if extra_info and 'clicked_idx' in extra_info:
        target = extra_info.get('clicked_idx')
    elif ground_truth:
        try:
            target = int(str(ground_truth).strip())
        except ValueError:
            match = re.search(r'(\d+)', str(ground_truth))
            if match:
                target = int(match.group(1))

    if target is None:
        return 0.0

    if predicted == target:
        return 1.0

    if extra_info and 'labels' in extra_info:
        labels = extra_info.get('labels', [])
        if 0 < predicted <= len(labels) and labels[predicted - 1] == 1:
            return 0.5

    if extra_info and 'categories' in extra_info:
        categories = extra_info.get('categories', [])
        if (0 < predicted <= len(categories) and
            0 < target <= len(categories)):
            pred_cat = categories[predicted - 1]
            target_cat = categories[target - 1]
            if pred_cat and target_cat and pred_cat == target_cat:
                return 0.3

    return 0.0


def compute_score_mind_cot_auc(data_source, solution_str, ground_truth, extra_info=None):
    """AUC reward (legacy single-answer). Use cot_prob_auc for new format."""
    if not extra_info or 'labels' not in extra_info:
        return compute_score_mind_cot_binary(data_source, solution_str, ground_truth, extra_info)

    labels = extra_info.get('labels', [])
    num_candidates = extra_info.get('num_candidates', len(labels))

    predicted = extract_cot_answer(solution_str, num_candidates)
    if predicted is None:
        return 0.0

    if 0 < predicted <= len(labels):
        if labels[predicted - 1] == 1:
            return 1.0

    return 0.0


def compute_score_mind_cot_margin(data_source, solution_str, ground_truth, extra_info=None):
    """Margin reward (legacy single-answer). Use cot_prob_margin for new format."""
    num_candidates = None
    labels = []
    if extra_info:
        num_candidates = extra_info.get('num_candidates')
        labels = extra_info.get('labels', [])

    predicted = extract_cot_answer(solution_str, num_candidates)
    if predicted is None:
        return -0.5

    target = None
    if extra_info and 'clicked_idx' in extra_info:
        target = extra_info.get('clicked_idx')
    elif ground_truth:
        try:
            target = int(str(ground_truth).strip())
        except ValueError:
            match = re.search(r'(\d+)', str(ground_truth))
            if match:
                target = int(match.group(1))

    if target and predicted == target:
        return 1.0

    if labels and 0 < predicted <= len(labels):
        if labels[predicted - 1] == 1:
            return 0.5
        else:
            return -0.3

    return 0.0


def compute_score_mind_cot_format(data_source, solution_str, ground_truth, extra_info=None):
    """
    Format reward updated for new <think>/<answer> format.
    Also backward compatible with old "Answer: X" format.
    """
    if not solution_str:
        return 0.0

    text = str(solution_str).strip()
    fmt = _check_cot_format(text)

    if fmt['good_format']:
        # New format detected — use prob-based AUC reward + format bonus
        base = compute_score_mind_cot_prob_auc(data_source, solution_str, ground_truth, extra_info)
        return min(1.0, base + 0.1)
    else:
        # Fallback to legacy format check
        has_reasoning = len(text) > 20
        has_explicit_answer = bool(re.search(r'[Aa]nswer\s*:\s*\d+', text))
        good_legacy = has_reasoning and has_explicit_answer

        base_reward = compute_score_mind_cot_binary(data_source, solution_str, ground_truth, extra_info)

        if base_reward == 1.0:
            return 1.0 if good_legacy else 0.8
        else:
            return 0.1 if good_legacy else 0.0
