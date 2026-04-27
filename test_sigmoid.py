import numpy as np

def orig(yes_no_logits):
    diff = yes_no_logits[:, 1] - yes_no_logits[:, 0]
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-diff))

def in_place(yes_no_logits):
    diff = yes_no_logits[:, 0] - yes_no_logits[:, 1]
    with np.errstate(over="ignore"):
        np.exp(diff, out=diff)
        diff += 1.0
        np.reciprocal(diff, out=diff)
        return diff

logits = np.array([[10.0, 5.0], [-2.0, 3.0]], dtype=np.float32)

print("orig:", orig(logits))
print("in_place:", in_place(logits.copy()))
