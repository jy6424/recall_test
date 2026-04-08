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

    # Save ground truth if available
    if "neighbors" in f:
        gt_file = f"groundtruth_coco.txt"
        np.savetxt(gt_file, f["neighbors"][:], fmt="%d")
        print(f"Ground truth saved to {gt_file}")

    f.close()
    print("Done!")
    print(f"  Insert: {insert_file} ({n_train} rows, {dim}-dim)")
    print(f"  Query:  {query_file} ({n_test} queries, k={k})")

if __name__ == "__main__":
    main()
