## 2025-02-28 - [Numpy Broadcasting Optimization]
**Learning:** Using `np.expand_dims` and native broadcasting is significantly faster (~4x speedup) and more memory-efficient than explicitly allocating expanded arrays using `np.tile`. Additionally, matching the `dtype` of masks to the `dtype` of the target arrays prevents implicit type promotion overhead during element-wise multiplication.
**Action:** Always prefer NumPy broadcasting over explicit tiling, and proactively cast masks to the appropriate float/int dtype before multiplying with target arrays.
