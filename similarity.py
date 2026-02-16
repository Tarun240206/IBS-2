import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def remove_redundancy(df, seq_col="Sequence", k=6, threshold=0.95):

    df = df.reset_index(drop=True)

    sequences = df[seq_col].astype(str)

    def get_kmers(seq):
        return " ".join([seq[i:i+k] for i in range(len(seq)-k+1)])

    kmer_data = sequences.apply(get_kmers)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(kmer_data)

    sim_matrix = cosine_similarity(X)

    remove_idx = set()
    n = sim_matrix.shape[0]

    for i in range(n):
        if i in remove_idx:
            continue

        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                remove_idx.add(j)

    cleaned_df = df.drop(index=list(remove_idx)).reset_index(drop=True)

    return cleaned_df


file_path = "final_3000_promoters.xlsx"
df = pd.read_excel(file_path)


df["Species"] = df["Species"].astype(str).str.strip()


rice_df = df[df["Species"] == "Rice"].reset_index(drop=True)
arab_df = df[df["Species"] == "Arabidopsis"].reset_index(drop=True)


if rice_df.empty or arab_df.empty:
    raise ValueError("Rice or Arabidopsis data missing.")


rice_cleaned = remove_redundancy(rice_df)
arab_cleaned = remove_redundancy(arab_df)


final_df = pd.concat([rice_cleaned, arab_cleaned], ignore_index=True)


output_file = "cleaned_rice_arab_single_sheet.xlsx"


final_df.to_excel(output_file, index=False)


print("Rice Original :", len(rice_df))
print("Rice Cleaned  :", len(rice_cleaned))

print("Arab Original:", len(arab_df))
print("Arab Cleaned :", len(arab_cleaned))

print("Final Total  :", len(final_df))

print("Saved as:", output_file)