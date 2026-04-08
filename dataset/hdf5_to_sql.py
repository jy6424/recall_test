#!/usr/bin/env python3
"""Convert ann-benchmarks HDF5 file to SQL insert/query files."""

import h5py
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 hdf5_to_sql.py <file.hdf5> [k] [max_inserts]")
        print("  k           = number of neighbors for top_k (default: 10)")
        print("  max_inserts = limit insert count (default: all)")
        sys.exit(1)

    hdf5_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_inserts = int(sys.argv[3]) if len(sys.argv) > 3 else None

    base = os.path.splitext(os.path.basename(hdf5_path))[0]

    f = h5py.File(hdf5_path, "r")
    print(f"Keys: {list(f.keys())}")

    train = f["train"]
    test = f["test"]
    dim = train.shape[1]
    n_train = train.shape[0] if max_inserts is None else min(train.shape[0], max_inserts)
    n_test = test.shape[0]

    print(f"Train: {train.shape} ({n_train} will be used)")
    print(f"Test:  {test.shape}")
    print(f"Dim:   {dim}, k={k}")

    if "neighbors" in f:
        print(f"Ground truth neighbors: {f['neighbors'].shape}")

    # Generate insert SQL
    insert_file = f"insert100k_coco.sql"
    print(f"Writing {insert_file} ...")
    with open(insert_file, "w") as out:
        out.write("PRAGMA journal_mode=WAL;\n")
        out.write(f"CREATE TABLE vectors(id INTEGER PRIMARY KEY, embedding F32_BLOB({dim}));\n")
        out.write("CREATE INDEX vec_idx ON vectors(libsql_vector_idx(embedding));\n")
        for i in range(n_train):
            vec = train[i]
            v = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            out.write(f"INSERT INTO vectors VALUES ({i+1}, vector32('{v}'));\n")

    # Generate query SQL
    query_file = f"query10k_coco.sql"
    print(f"Writing {query_file} ...")
    with open(query_file, "w") as out:
        for vec in test:
            v = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            out.write(f"SELECT id FROM vector_top_k('vec_idx', vector32('{v}'), {k});\n")

    # Recompute ground truth for the subset
    print(f"Computing ground truth for {n_train} vectors, {n_test} queries (k={k}) ...")
    train_np = train[:n_train]
    test_np = test[:]
    # Normalize for cosine similarity
    train_norm = train_np / np.linalg.norm(train_np, axis=1, keepdims=True)
    test_norm = test_np / np.linalg.norm(test_np, axis=1, keepdims=True)
    # Compute cosine similarity in batches to save memory
    neighbors = np.zeros((n_test, k), dtype=np.int32)
    batch_size = 100
    for i in range(0, n_test, batch_size):
        end = min(i + batch_size, n_test)
        sims = test_norm[i:end] @ train_norm.T  # (batch, n_train)
        topk = np.argpartition(-sims, k, axis=1)[:, :k]
        for j in range(end - i):
            idx = topk[j]
            idx = idx[np.argsort(-sims[j][idx])]
            neighbors[i + j] = idx
        if (i // batch_size) % 10 == 0:
            print(f"  {i}/{n_test} queries done")
    # Save as 1-indexed IDs (matching SQL INSERT IDs)
    gt_file = f"groundtruth_coco.txt"
    np.savetxt(gt_file, neighbors + 1, fmt="%d")
    print(f"Ground truth saved to {gt_file} (shape: {neighbors.shape})")

    f.close()
    print("Done!")
    print(f"  Insert: {insert_file} ({n_train} rows, {dim}-dim)")
    print(f"  Query:  {query_file} ({n_test} queries, k={k})")

if __name__ == "__main__":
    main()
