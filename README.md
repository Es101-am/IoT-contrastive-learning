# 🧠 Contrastive Learning for Multi-Sensor IoT Data

This repository implements a contrastive learning framework to align observations from two different agents (or sensor types) — a thermostat (Agent 1) and power+humidity sensors (Agent 2). The model learns to project different sensor views of the same environment into a shared embedding space using cosine similarity and contrastive loss.

---

## 🔍 Overview

- **Agent 1 (Thermostat)** observes:
  - Set temperature
  - Current temperature
  - Room humidity
  - A shared pair token

- **Agent 2 (Power+Humidity)** observes:
  - Total power consumption
  - HVAC contribution
  - House ID (normalized)
  - Occupancy (binary)
  - A shared pair token

Both views are generated using synthetic logic that simulates realistic HVAC and energy behavior based on time of day, comfort gaps, and occupancy.

---

## 🧪 Training Objective

Using contrastive learning, we train two separate encoders (`encoder1`, `encoder2`) such that:
- Matching pairs from the same timestamp are **pulled together** in embedding space
- Mismatched pairs from other time steps are **pushed apart**

This is done using a symmetric InfoNCE-like loss based on cosine similarity.

---

## 📁 Project Structure

- `PairedSensorDataset`: Generates synthetic, normalized, and aligned observations from both agents.
- `AgentEncoder`: Two-layer encoder + projector with L2 normalization to embed sensor data.
- `train_contrastive()`: Core training loop using cross-entropy loss over cosine similarities.
- `Temperature Annealing`: Starts with a soft contrastive objective and tightens over epochs.

---

## 🔧 Use Cases for Trained Encoders

After training, you now have:

- `encoder1(x1)`: maps **thermostat data** → embedding  
- `encoder2(x2)`: maps **power + humidity data** → embedding  
✅ These embeddings are **aligned**, i.e., close if they represent the same underlying environmental condition.

---

### ✅ 1. Sensor Fusion for Decision-Making
Use either (or both) encoders to:
- Represent **multi-view states** in a reinforcement learning (RL) setup
- Combine incomplete sensor streams for robust control or predictions

**Example**:  
> Use either encoder’s output as state input to a policy network for smart HVAC control.

---

### ✅ 2. Missing Modality Inference
If one sensor fails (e.g., thermostat disconnected), you can:
- Use `encoder2(x2)` to get an **approximate latent representation**
- Feed it into downstream models trained using `encoder1(x1)` — because they're aligned

---

### ✅ 3. Anomaly Detection
Normal pairs map to **close embeddings**. If:  
- \( ||encoder1(x_1) - encoder2(x_2)|| \) is **large**, it's likely:
  - Sensors are **not consistent**
  - Something is **wrong** in the system (anomaly)

**Example**:  
> HVAC reports cooling, but thermostat reports temp rising → possible fault.

---

### ✅ 4. Cross-Modal Retrieval
Use embeddings to **search across sensor modalities**:
- Query with thermostat state → find similar past situations in power/humidity history  
- Or the other way around

**Example**:  
> “When did we last consume similar power while set temp was 22.5°C?”

---

### ✅ 5. Downstream Supervised Learning
Use embeddings as inputs to a classifier or regressor.

Instead of raw sensor features:
```python
z1 = encoder1(x1)
pred = classifier(z1)


