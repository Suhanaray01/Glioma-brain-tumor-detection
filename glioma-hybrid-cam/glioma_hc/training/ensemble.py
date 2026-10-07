from __future__ import annotations

import numpy as np


def hard_vote(prob_matrix: np.ndarray) -> tuple[str, int, list[int]]:
    vote_counts = np.sum(prob_matrix > 0.5, axis=0)
    winner = int(np.argmax(vote_counts))
    return ("glioma" if winner == 1 else "normal", int(np.max(vote_counts)), vote_counts.tolist())


def soft_vote(prob_matrix: np.ndarray) -> tuple[str, np.ndarray]:
    summed = np.sum(prob_matrix, axis=0)
    winner = int(np.argmax(summed))
    return ("glioma" if winner == 1 else "normal", summed)


def majority_vote(probabilities: list[np.ndarray], threshold: float = 0.5) -> dict:
    arr = np.asarray(probabilities)
    hard_label, votes, counts = hard_vote(arr)
    soft_label, soft_probs = soft_vote(arr)
    return {"hard": {"label": hard_label, "votes": votes, "counts": counts}, "soft": {"label": soft_label, "probs": soft_probs.tolist(), "threshold": threshold}}
