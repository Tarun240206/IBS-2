import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner
from math import log2


df = pd.read_excel("cleaned_rice_arab_single_sheet.xlsx")

df["Species"] = df["Species"].astype(str).str.strip()

arab_df = df[df["Species"] == "Arabidopsis"].copy().reset_index(drop=True)
rice_df = df[df["Species"] == "Rice"].copy().reset_index(drop=True)


arab_reference = arab_df.iloc[0]["Sequence"]
rice_reference = rice_df.iloc[0]["Sequence"]


aligner = PairwiseAligner()

aligner.mode = "global"
aligner.match_score = 1
aligner.mismatch_score = -1
aligner.open_gap_score = -1
aligner.extend_gap_score = -1


def compute_alignment_features(seq, reference):

    alignments = aligner.align(seq, reference)
    best = alignments[0]

    score = best.score

    aligned1, aligned2 = best.aligned

    matches = 0
    length = 0

    for (s1, e1), (s2, e2) in zip(aligned1, aligned2):

        part1 = seq[s1:e1]
        part2 = reference[s2:e2]

        for a, b in zip(part1, part2):
            if a == b:
                matches += 1
            length += 1

    identity = (matches / length) * 100 if length > 0 else 0

    return score, identity


def cpg_ratio(seq):

    c = seq.count("C")
    g = seq.count("G")
    cg = seq.count("CG")
    n = len(seq)

    if c * g == 0:
        return 0

    return (cg * n) / (c * g)


def motif_position(seq, motif="TATAAA"):

    pos = seq.find(motif)

    return pos if pos != -1 else np.nan


def shannon_entropy(seq):

    n = len(seq)
    entropy = 0

    for base in ["A", "T", "G", "C"]:

        p = seq.count(base) / n

        if p > 0:
            entropy -= p * log2(p)

    return entropy


arab_scores = []
arab_ids = []
arab_cpg = []
arab_motif = []
arab_entropy = []


for seq in arab_df["Sequence"]:

    score, identity = compute_alignment_features(seq, arab_reference)

    arab_scores.append(score)
    arab_ids.append(identity)
    arab_cpg.append(cpg_ratio(seq))
    arab_motif.append(motif_position(seq))
    arab_entropy.append(shannon_entropy(seq))


arab_df["Alignment_Score"] = arab_scores
arab_df["Sequence_Identity_%"] = arab_ids
arab_df["CpG_Ratio"] = arab_cpg
arab_df["Motif_Position"] = arab_motif
arab_df["arab_entropy"] = arab_entropy
arab_df["rice_entropy"] = np.nan


rice_scores = []
rice_ids = []
rice_cpg = []
rice_motif = []
rice_entropy = []


for seq in rice_df["Sequence"]:

    score, identity = compute_alignment_features(seq, rice_reference)

    rice_scores.append(score)
    rice_ids.append(identity)
    rice_cpg.append(cpg_ratio(seq))
    rice_motif.append(motif_position(seq))
    rice_entropy.append(shannon_entropy(seq))


rice_df["Alignment_Score"] = rice_scores
rice_df["Sequence_Identity_%"] = rice_ids
rice_df["CpG_Ratio"] = rice_cpg
rice_df["Motif_Position"] = rice_motif
rice_df["rice_entropy"] = rice_entropy
rice_df["arab_entropy"] = np.nan


final_df = pd.concat([arab_df, rice_df], ignore_index=True)


final_df["mean_rice_entropy"] = rice_df["rice_entropy"].mean()
final_df["mean_arab_entropy"] = arab_df["arab_entropy"].mean()


sample_size = 100

rice_sample = rice_df.sample(n=sample_size, random_state=42)
arab_sample = arab_df.sample(n=sample_size, random_state=42)


cross_ids = []
cross_scores = []


for rseq in rice_sample["Sequence"]:
    for aseq in arab_sample["Sequence"]:

        score, identity = compute_alignment_features(rseq, aseq)

        cross_scores.append(score)
        cross_ids.append(identity)


final_df["mean_identity"] = np.mean(cross_ids)
final_df["mean_alignment"] = np.mean(cross_scores)


output_file = "promoters_with_all_features_plus_crossmeans.xlsx"

final_df.to_excel(output_file, index=False)

print("Saved:", output_file)