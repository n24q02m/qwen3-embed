## 2024-05-24 - [Optimize mean_pooling]
**Learning:** In `mean_pooling`, computing masked sequence embeddings using `np.matmul` with a mask of shape `(Batch, 1, Seq)` (e.g., `np.matmul(mask[:, np.newaxis, :], input_array).squeeze(1)`) is significantly faster (~6x) and more memory-efficient than using `np.expand_dims` and `np.sum` for broadcasting, while explicitly casting the mask to the input array's dtype avoids type promotion overhead.
**Action:** Replace `np.expand_dims` and `np.sum` with `np.matmul` when calculating the sum of masked token embeddings.
