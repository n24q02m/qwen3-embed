## 2025-03-01 - [Optimize mean_pooling in utils.py]
**Learning:** In NumPy, computing masked sequence embeddings using `np.matmul(mask[:, np.newaxis, :], input_array).squeeze(1)` is significantly faster (~6x in benchmarks) and more memory efficient than using `np.expand_dims` and `np.sum` for broadcasting `(input_array * input_mask_expanded)`.
**Action:** Use `np.matmul` when calculating weighted sums of embeddings instead of expanding masks and relying on element-wise multiplication followed by sum reduction.
