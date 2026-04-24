# # 🧠 Self-Pruning Neural Network

## 📌 Overview
This project implements a neural network that **prunes itself during training**.  
Instead of removing weights after training, the model learns which connections are unnecessary using **learnable gates** and **L1 regularization**.

---

## ⚙️ Methodology

### 🔹 Prunable Layer
A custom `PrunableLinear` layer is used instead of `nn.Linear`.

- Each weight has a **gate score**
- Gate values are computed using:
  
  `gate = sigmoid(gate_score)`

- Final weights:
  
  `pruned_weight = weight × gate`

👉 If gate → 0 → connection removed  
👉 If gate → 1 → connection retained  

---

### 🔹 Loss Function

`Total Loss = CrossEntropyLoss + λ × SparsityLoss`

- **SparsityLoss** = sum of all gate values  
- L1 regularization forces many gates → 0  
- λ controls sparsity vs accuracy trade-off  

---

## 🏋️ Training

- Dataset: CIFAR-10  
- Optimizer: Adam  
- Model: Fully connected network using prunable layers  
- Multiple λ values tested  

---

## 📊 Results

| λ Value | Accuracy | Sparsity |
|--------|---------|---------|
| Low    | High    | Low     |
| Medium | Balanced| Medium  |
| High   | Lower   | High    |

👉 Higher λ increases pruning but may reduce accuracy  

---

## 📈 Key Insight
L1 regularization pushes gate values toward **zero**, effectively removing unimportant connections.  
This allows the network to **learn an optimal sparse structure automatically**.

---

## 🚀 Conclusion
The model successfully:
- Learns to prune itself  
- Reduces unnecessary parameters  
- Maintains reasonable performance  

This approach is useful for building **efficient and scalable AI systems**.

---

## 💻 Tech Stack
- Python  
- PyTorch  
- Torchvision  

---

## ▶️ Run

```bash
pip install torch torchvision
python main.py
