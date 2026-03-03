## 2024-05-24 - Speeding up mean pooling
**Learning:** In numpy, computing masked sequence embeddings for mean pooling using `np.matmul` with a mask of shape `(Batch, 1, Seq)` (e.g., `np.matmul(mask[:, np.newaxis, :], input_array).squeeze(1)`) is significantly faster (~18x) and more memory-efficient than using `np.expand_dims` and `np.sum` for broadcasting.
**Action:** Always prefer `np.matmul` over `np.expand_dims` + `np.sum` when applying a 2D mask (like an attention mask) to a 3D tensor across the sequence dimension.
