---
license: apache-2.0
---

# 💻 Dataset Usage
Run the following command to load the testing set (1,273 examples):
```python
from datasets import load_dataset

dataset = load_dataset("shuyuej/MedQA-USMLE-Benchmark", split="test")
print(dataset)
```
