# %% [markdown]
# # SmartFarm AI – Week 2 Concepts Demo
# 
# Theme: Machine Learning Meets the UN Sustainable Development Goals (SDGs)
# 
# SDG Focus: SDG 2 – Zero Hunger
# 
# This interactive script demonstrates Week 2 ML concepts using lightweight, synthetic data:
# - Supervised Learning (CNN preview) – crop disease detection
# - Unsupervised Learning – KMeans clustering on synthetic NDVI features
# - Neural Networks – yield prediction (ensemble + dense NN)
# - Reinforcement Learning – smart irrigation environment step-through
# - NLP – sentiment analysis (transformers/TextBlob fallback)
# 
# Run cells with VS Code Python: use the Run Cell button above each cell.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Visualization config
plt.rcParams["figure.figsize"] = (7, 4)

# %% [markdown]
# ## Supervised Learning (CNN preview)
# We'll import the model class and build the architecture (no training to keep it fast).

# %%
try:
    from src.crop_disease_detection import CropDiseaseDetector

    detector = CropDiseaseDetector()
    model = detector.build_model()
    print(f"CNN model built with {model.count_params():,} parameters")

    # Dummy inference on a random image-like array
    dummy_image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    # Preprocess expects a path; we'll adapt quickly for demonstration
    x = np.expand_dims(dummy_image / 255.0, axis=0)
    y = model.predict(x, verbose=0)
    print("Dummy prediction vector (sum=1):", np.round(y[0], 3))
except Exception as e:
    print("CNN preview skipped (install TensorFlow to enable). Reason:", e)

# %% [markdown]
# ## Unsupervised Learning – KMeans clustering of NDVI features
# Create synthetic NDVI stats per plot and cluster into 3 groups.

# %%
np.random.seed(42)
ndvi_avg = np.random.beta(2, 1, 300) * 0.8 + 0.2
ndvi_var = np.random.rand(300) * 0.05
lai = np.random.exponential(2, 300)
X = np.vstack([ndvi_avg, ndvi_var, lai]).T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

plt.scatter(ndvi_avg, lai, c=labels, cmap="viridis", alpha=0.7)
plt.xlabel("NDVI Avg")
plt.ylabel("LAI")
plt.title("KMeans clusters of plots (NDVI vs LAI)")
plt.show()

# %% [markdown]
# ## Neural Networks – Yield prediction (synthetic data)
# We'll use the provided system to generate synthetic data and train models quickly.

# %%
try:
    from src.yield_prediction import YieldPredictionSystem

    ys = YieldPredictionSystem()
    data = ys.create_synthetic_data(n_samples=1000)
    results, _ = ys.train_ensemble(data)
    print("Model results (RMSE):", {k: round(v['rmse'], 3) for k, v in results.items()})
except Exception as e:
    print("Yield prediction demo skipped (install scikit-learn/TF/XGBoost/LightGBM to enable). Reason:", e)

# %% [markdown]
# ## Reinforcement Learning – Smart irrigation (short rollout)

# %%
try:
    from src.smart_irrigation import IrrigationEnvironment

    env = IrrigationEnvironment()
    state = env.reset()
    # Random policy for a few steps
    rewards = []
    for _ in range(5):
        action = np.array([np.random.uniform(0, 20)], dtype=np.float32)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        state = next_state
    print("5-step random policy rewards:", np.round(rewards, 2))
except Exception as e:
    print("RL demo skipped (install gym dependencies to enable). Reason:", e)

# %% [markdown]
# ## NLP – Sentiment analysis demo (with graceful fallback)

# %%
from typing import List, Dict

try:
    from src.market_intelligence import MarketIntelligenceSystem
    mis = MarketIntelligenceSystem()
    headlines: List[str] = [
        "Rice export demand surges amid global shortages",
        "Monsoon rains decline; crop output may decrease",
        "Government announces support price increase for wheat",
        "Fertilizer costs drop as supply improves",
        "Pest outbreak reported in northern region"
    ]
    sentiments: List[Dict] = mis.analyze_sentiment(headlines)
    print("Headline sentiments (label, polarity):")
    for s in sentiments:
        print("-", s["label"], f"{s['polarity']:+.2f}", "::", s["text"]) 
except Exception as e:
    print("NLP demo skipped. Reason:", e)

# %% [markdown]
# ## Impact & Metrics (illustrative)

# %%
metrics = {
    "Yield Improvement (%)": 20,
    "Water Savings (%)": 40,
    "Income Increase (%)": 25,
}
plt.bar(metrics.keys(), metrics.values(), color=["#4CAF50", "#2196F3", "#FF9800"])
plt.ylim(0, 60)
plt.title("Expected Impact Metrics")
plt.show()
