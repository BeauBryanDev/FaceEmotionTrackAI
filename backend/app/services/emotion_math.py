import numpy as np

def compute_entropy(probabilities: list[float]) -> float:
    """
    Compute Shannon entropy for a probability distribution.
    """

    p = np.array(probabilities)

    # avoid log(0)
    p = p[p > 0]

    entropy = -np.sum(p * np.log(p))

    return float(entropy)
