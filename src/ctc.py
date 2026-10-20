"""CTC helpers: greedy decoding and edit distance (for character error rate)."""
import numpy as np


def greedy_decode(logits, charset: str):
    """logits: (T, B, K) tensor with class 0 = blank. Returns a list of B strings."""
    best = logits.argmax(2).T.cpu().numpy()  # B, T
    results = []
    for seq in best:
        prev, chars = -1, []
        for k in seq:
            if k != prev and k != 0:
                chars.append(charset[k - 1])
            prev = k
        results.append("".join(chars))
    return results


def edit_distance(a: str, b: str) -> int:
    dp = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return int(dp[-1])
