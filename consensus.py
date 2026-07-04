"""
consensus.py — Multiple Sequence Alignment Consensus Engine

Combines OCR outputs from multiple engines/variants into a single
best-guess string using:

1. Character similarity matrix (confusable pairs like 0/o, 5/s)
2. Asymmetric Medoid selection (prefers longer strings)
3. MSA backbone correction (recovers dropped characters)
4. Per-character case voting (EasyOCR primary, ddddocr fallback)
5. Confidence scoring (inter-read agreement metric)
"""

from __future__ import annotations

import re
from collections import Counter


# ── Character Similarity ─────────────────────────────────────

SIMILAR_CHARS = {
    ('0', 'o'), ('1', 'l'), ('1', 'i'), ('i', 'l'),
    ('5', 's'), ('2', 'z'), ('7', 'z'), ('c', 'e'),
    ('@', 'c'), ('@', 'a'), ('8', 'b'), ('6', 'b'),
    ('9', 'g'), ('9', 'q'), ('u', 'v'), ('n', 'm'),
    ('d', '0'), ('d', 'o'),  # D/0 confusion on noisy CAPTCHAs
}


def char_dist(c1: str, c2: str) -> float:
    """Distance between two characters: 0.0 = same, 0.5 = similar, 1.0 = different."""
    c1, c2 = c1.lower(), c2.lower()
    if c1 == c2:
        return 0.0
    if (c1, c2) in SIMILAR_CHARS or (c2, c1) in SIMILAR_CHARS:
        return 0.5
    return 1.0


# ── Asymmetric Levenshtein ───────────────────────────────────

def _lev_dist_asym(candidate: str, target: str) -> float:
    """
    Asymmetric Levenshtein distance.

    Deletion from candidate is cheap  (0.2) — having extra chars is ok.
    Insertion into candidate is costly (1.5) — missing chars is bad.
    This biases the Medoid toward the longest plausible string.
    """
    m, n = len(candidate), len(target)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * 0.2
    for j in range(n + 1):
        dp[0][j] = j * 1.5
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = char_dist(candidate[i - 1], target[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 0.2,        # delete from candidate
                dp[i][j - 1] + 1.5,        # insert into candidate
                dp[i - 1][j - 1] + sub,    # substitute
            )
    return dp[m][n]


# ── Symmetric Levenshtein (for alignment) ────────────────────

def _lev_align(a: str, b: str) -> list[tuple]:
    """
    Standard Levenshtein alignment returning (char_a, char_b) pairs.
    None in either position means a gap (insertion / deletion).
    """
    m, n = len(a), len(b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = char_dist(a[i - 1], b[j - 1])
            dp[i][j] = min(dp[i - 1][j] + 1.0,
                           dp[i][j - 1] + 1.0,
                           dp[i - 1][j - 1] + sub)
    # Traceback
    pairs = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = char_dist(a[i - 1], b[j - 1])
            if dp[i][j] == dp[i - 1][j - 1] + sub:
                pairs.append((a[i - 1], b[j - 1]))
                i -= 1; j -= 1
                continue
        if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1.0):
            pairs.append((a[i - 1], None))
            i -= 1
        else:
            pairs.append((None, b[j - 1]))
            j -= 1
    return list(reversed(pairs))


# ── MSA Consensus ────────────────────────────────────────────

def medoid_consensus(reads: list[str]) -> str:
    """
    Two-phase consensus:
      1. Length-mode filter — keep only reads matching the most common
         length to eliminate phantom characters from noise.
      2. Asymmetric Medoid — pick the read that minimises total
         distance to all others (biased toward longer strings).
      3. MSA correction — align every read to the backbone and
         vote on each position + splice in consistently-inserted
         characters that ≥2 reads agree on.
    """
    if not reads:
        return ""

    # Phase 0: Length-mode filter
    # When noise causes phantom characters, some reads will be longer
    # than others. Filter to the most common length if it has a clear
    # plurality (≥40% of reads). This prevents noise-induced extra
    # characters from surviving into the consensus.
    len_counts = Counter(len(r) for r in reads)
    modal_len, modal_count = len_counts.most_common(1)[0]
    if modal_count >= len(reads) * 0.4:
        filtered = [r for r in reads if len(r) == modal_len]
        if len(filtered) >= 3:  # need enough reads for meaningful consensus
            reads = filtered

    # Phase 1: Asymmetric Medoid
    best_read = reads[0]
    best_dist = float("inf")
    for c in reads:
        dist = sum(_lev_dist_asym(c, t) for t in reads)
        if dist < best_dist:
            best_dist = dist
            best_read = c

    backbone = best_read

    # Phase 2: MSA correction
    aligned_chars: list[list[str]] = [[] for _ in backbone]
    inserted_strings: dict[int, list[str]] = {i: [] for i in range(-1, len(backbone))}

    for r in reads:
        m, n = len(backbone), len(r)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        path = [[None] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i * 1.0
            path[i][0] = ("del", i - 1, 0) if i > 0 else None
        for j in range(n + 1):
            dp[0][j] = j * 1.0
            path[0][j] = ("ins", 0, j - 1) if j > 0 else None

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c_back, c_read = backbone[i - 1], r[j - 1]
                if c_back == c_read:
                    sub_cost = 0.0
                elif (c_back, c_read) in SIMILAR_CHARS or (c_read, c_back) in SIMILAR_CHARS:
                    sub_cost = 0.5
                else:
                    sub_cost = 1.0

                cost_del = dp[i - 1][j] + 1.0
                cost_ins = dp[i][j - 1] + 1.0
                cost_sub = dp[i - 1][j - 1] + sub_cost

                min_c = min(cost_del, cost_ins, cost_sub)
                dp[i][j] = min_c
                if min_c == cost_sub:
                    path[i][j] = ("sub", i - 1, j - 1)
                elif min_c == cost_ins:
                    path[i][j] = ("ins", i, j - 1)
                else:
                    path[i][j] = ("del", i - 1, j)

        # Traceback
        ops = []
        ci, cj = m, n
        while ci > 0 or cj > 0:
            op, pi, pj = path[ci][cj]
            if op == "sub":
                b_idx, r_idx = pi, pj
            elif op == "del":
                b_idx, r_idx = pi, -1
            elif op == "ins":
                b_idx, r_idx = max(pi - 1, -1), pj
            ops.append((op, b_idx, r_idx))
            ci, cj = pi, pj
        ops.reverse()

        # Collect aligned chars and insertions
        current_ins: list[str] = []
        last_bb_idx = -1

        for op, b_idx, r_idx in ops:
            if op == "sub":
                if current_ins:
                    inserted_strings[last_bb_idx].append("".join(current_ins))
                    current_ins = []
                aligned_chars[b_idx].append(r[r_idx])
                last_bb_idx = b_idx
            elif op == "ins":
                if b_idx != last_bb_idx:
                    if current_ins:
                        inserted_strings[last_bb_idx].append("".join(current_ins))
                        current_ins = []
                    last_bb_idx = b_idx
                current_ins.append(r[r_idx])
            elif op == "del":
                if current_ins:
                    inserted_strings[last_bb_idx].append("".join(current_ins))
                    current_ins = []
                last_bb_idx = b_idx

        if current_ins:
            inserted_strings[last_bb_idx].append("".join(current_ins))

    # Build final string from votes
    def _winner(lst: list, min_votes: int, default: str = "") -> str:
        if not lst:
            return default
        c = Counter(lst)
        top_count = c.most_common(1)[0][1]
        if top_count < min_votes:
            return default
        candidates = [k for k, v in c.items() if v == top_count]
        return default if default in candidates else candidates[0]

    final: list[str] = []

    ins = _winner(inserted_strings[-1], min_votes=2)
    if ins:
        final.append(ins)

    for i in range(len(backbone)):
        winner = _winner(aligned_chars[i], min_votes=1, default=backbone[i])
        final.append(winner)
        ins = _winner(inserted_strings[i], min_votes=2)
        if ins:
            final.append(ins)

    return "".join(final)


# ── Per-character case voting ────────────────────────────────

def per_char_case_vote(
    base_reads: list[str],
    beta_reads: list[str],
    easy_text: str,
    log,
    easy_conf: float = 0.0,
) -> tuple[str, float]:
    """
    Determine final characters with correct casing.

    Uses base+beta reads for character identity (MSA consensus),
    and EasyOCR as the PRIMARY case source.

    Returns (final_text, confidence) where confidence is 0.0–1.0.
    """
    if not base_reads:
        return "", 0.0

    # 1. MSA consensus over all ddddocr reads
    all_reads = base_reads + beta_reads
    if not all_reads:
        return "", 0.0

    base_consensus = medoid_consensus(all_reads).lower()
    mode_len = len(base_consensus)
    if mode_len == 0:
        return "", 0.0

    # 2. Align EasyOCR text to base consensus
    easy_aligned: list[str | None] = [None] * mode_len
    easy_clean = re.sub(r"[^\x21-\x7E]", "", easy_text) if easy_text else ""
    if easy_clean:
        pairs = _lev_align(base_consensus, easy_clean)
        pos = 0
        for c_base, c_easy in pairs:
            if c_base is None:
                continue
            if pos < mode_len:
                easy_aligned[pos] = c_easy
            pos += 1

    # 3. Align ALL reads to collect case votes per position
    votes_per_pos: list[list[str]] = [[] for _ in range(mode_len)]
    for r in all_reads:
        pairs = _lev_align(base_consensus, r)
        pos = 0
        for c_base, c_read in pairs:
            if c_base is None:
                continue
            if pos < mode_len and c_read is not None:
                votes_per_pos[pos].append(c_read)
            pos += 1

    # 4. Per-character case determination
    final_chars: list[str] = []
    position_agreements: list[float] = []

    for i in range(mode_len):
        base_char = base_consensus[i]

        # Confidence: what fraction of reads agree on this character?
        if votes_per_pos[i]:
            vote_counts = Counter(c.lower() for c in votes_per_pos[i])
            top_count = vote_counts.most_common(1)[0][1]
            position_agreements.append(top_count / len(votes_per_pos[i]))
        else:
            position_agreements.append(0.0)

        # Digits don't have case
        if base_char.isdigit():
            final_chars.append(base_char)
            continue

        # PRIMARY: EasyOCR case (when aligned character matches)
        easy_c = easy_aligned[i]
        if easy_c is not None and char_dist(easy_c, base_char) < 1.0:
            final_c = base_char.upper() if easy_c.isupper() else base_char.lower()
            final_chars.append(final_c)
            continue

        # SECONDARY: Bag-of-Characters check in EasyOCR
        found_in_easy = False
        for ec in easy_clean:
            if char_dist(ec, base_char) < 1.0:
                final_chars.append(base_char.upper() if ec.isupper() else base_char.lower())
                found_in_easy = True
                break
        if found_in_easy:
            continue

        # FALLBACK: ddddocr case votes
        upper_n = sum(1 for c in votes_per_pos[i]
                      if char_dist(c, base_char) < 1.0 and c.isupper())
        lower_n = sum(1 for c in votes_per_pos[i]
                      if char_dist(c, base_char) < 1.0 and c.islower())

        if upper_n > 0 and lower_n == 0:
            final_chars.append(base_char.upper())
        elif upper_n > 0 and upper_n * 3 >= lower_n:
            final_chars.append(base_char.upper())
        else:
            final_chars.append(base_char.lower())

    result = "".join(final_chars)

    # 5. Symbol recovery — GUARDED
    # Only trust EasyOCR's special character when:
    #   (a) ddddocr also has ≥2 votes for a non-alphanumeric char at that position, OR
    #   (b) EasyOCR confidence is high (≥0.80)
    # This prevents low-confidence EasyOCR hallucinations (e.g. '-', '%') from
    # overriding unanimous ddddocr alphanumeric reads (e.g. 'A').
    for i in range(mode_len):
        easy_c = easy_aligned[i]
        if easy_c is not None and not easy_c.isalnum():
            if result[i].isalnum():
                # Check if ddddocr also saw special chars at this position
                dddd_special_votes = sum(
                    1 for c in votes_per_pos[i] if not c.isalnum()
                )
                if dddd_special_votes >= 2:
                    # ddddocr corroborates — trust the symbol
                    result = result[:i] + easy_c + result[i + 1:]
                elif easy_conf >= 0.80:
                    # EasyOCR is confident — trust it
                    result = result[:i] + easy_c + result[i + 1:]
                # else: ddddocr disagrees AND EasyOCR is low-confidence → keep ddddocr

    # 6. Compute overall confidence
    confidence = sum(position_agreements) / len(position_agreements) if position_agreements else 0.0

    log(f"  [vote] Base/Beta/Easy consensus → '{result}'  conf={confidence:.2f}")
    return result, confidence

