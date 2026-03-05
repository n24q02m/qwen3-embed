## 2024-05-24 - Optimize mean_pooling using np.matmul
**Learning:** In `mean_pooling`, computing masked sequence embeddings using `np.matmul` with a mask of shape `(Batch, 1, Seq)` is significantly faster (~5x) and more memory-efficient than using `np.expand_dims` and `np.sum` for broadcasting, while explicitly casting the mask to the input array's dtype avoids type promotion overhead.
**Action:** Use `np.matmul` instead of element-wise multiplication and summation over an axis when aggregating masked embeddings.
