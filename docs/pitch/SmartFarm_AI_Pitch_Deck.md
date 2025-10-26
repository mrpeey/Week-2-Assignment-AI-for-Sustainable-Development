# SmartFarm AI – Elevation Pitch Deck (SDG 2: Zero Hunger)

Version: 1.0  
Contact: Poulo Poulo/pppoulo@gmail.com

---

## 1. Title Slide
- SmartFarm AI: Practical AI for Smallholder Farmers
- Advancing UN SDG 2 – Zero Hunger
- One-line: "From reactive farming to proactive decisions—on any laptop."

---

## 2. Problem (Why Now?)
- 828M people face food insecurity; smallholders feed much of the world
- Farmers lack timely agronomy guidance and market signals
- Water scarcity + climate variability make irrigation risky
- Disease outbreaks can slash yields within weeks
- Fragmented tools; heavy AI stacks don’t run in low-resource contexts

---

## 3. Solution (What We Do)
SmartFarm AI is a modular decision-support toolkit that runs even without heavy ML installs:
- Detect crop diseases early from leaf photos (CV)
- Predict yields with confidence bands (ML + NN)
- Recommend water use with smart irrigation (RL)
- Decode market mood for better selling (NLP)
- Teach Week‑2 AI concepts with a friendly dashboard and notebook

---

## 4. Product (Demo View)
- Streamlit dashboard with 5 tabs: Disease, Irrigation, Yield, Market, Analytics
- FastAPI backend: /health, /yield/predict, /irrigation/recommend, /market/sentiment
- Windows one‑click scripts to launch
- Swagger UI at /docs for quick testing

[Placeholder: Insert dashboard screenshot]

---

## 5. How It Works (Tech)
- Supervised Learning (CNN) for leaf disease detection
- Neural Networks + Ensembles for yield (RF/XGBoost/NN)
- Reinforcement Learning (DQN) for irrigation scheduling
- NLP sentiment (VADER/TextBlob/Transformers)
- Unsupervised K‑Means to reveal management zones
- Graceful fallbacks: transparent heuristics when heavy libs absent

---

## 6. Why We Win (Differentiation)
- Works in "lightweight mode"—no GPU required
- Clear confidence/notes; explainable and safe defaults
- Education-ready: Week‑2 notebook, demos, tests
- Modular API enables easy integration
- Windows-friendly scripts reduce setup friction

---

## 7. Impact (SDG 2 Outcomes)
- +15–25% yield via better timing and disease control
- −30–50% water use through moisture-aware scheduling
- +20–30% income resilience with smarter market timing
- Community training via notebook and dashboard

---

## 8. Traction/Validation
- Automated CI runs API tests on each push
- Full demo runs in minutes on a typical laptop
- Fallback logic verified for low‑compute contexts
- Repo and docs ready for classrooms and pilots

---

## 9. Go-To-Market & Users
- Primary users: Smallholder farmers, extension officers, cooperatives
- Channels: NGOs, government programs, agtech integrators, universities
- Motion: Start with training workshops → community pilots → local champions

---

## 10. Business/Adoption Model (examples)
- Open-source core; paid support for deployments
- Partner-led pilots with NGO/government funding
- Tiered advisory: free basics + premium data integrations (sensors/NDVI)

---

## 11. Roadmap
- Data integrations: Sentinel‑2 NDVI, soil moisture sensors, weather APIs
- Models: Few‑shot disease detection on-device; meta-learning for varieties
- Product: Mobile app, offline bundles, multilingual UI, exports
- Governance: Feedback loops, impact dashboards, bias checks

---

## 12. Team & Advisors (placeholder)
- Add names/roles, domain experts, partner orgs

---

## 13. The Ask
- Partners for field pilots and local dataset collection
- Universities/NGOs to co-run training cohorts
- Contributors for data connectors and translations

---

## 14. Call to Action
- Try the dashboard or API locally (5 minutes)
- Use the notebook for hands-on learning
- Fork the repo and propose a pilot in your community

Repo: https://github.com/mrpeey/Week-2-Assignment-AI-for-Sustainable-Development

---

## Appendix A – 30‑Second Elevator Pitch
"SmartFarm AI is a practical, modular tool that helps smallholder farmers make better decisions across the crop cycle. It detects diseases from leaf images, predicts yields, optimizes irrigation with reinforcement learning, and analyzes market sentiment—all accessible via a simple dashboard and API. It’s designed for real constraints: it runs in lightweight mode without heavy installs and explains its recommendations. In short, it turns AI into timely, trustworthy advice—raising yields, saving water, and strengthening incomes for SDG 2."

---

## Appendix B – Live Demo Script (3–5 minutes)
1) Dashboard Overview: 10 seconds on tabs and purpose
2) Disease Detection: Upload image → mock result + treatment note
3) Irrigation: Show history chart; display today’s recommendation and reasoning
4) Yield: Confidence band and drivers; discuss how factors influence outcomes
5) Market: Trend line + sentiment result from headlines
6) End with impact slide and the call to action

---

## Appendix C – Visual Assets (placeholders)
- Screenshot: Overview metrics (yield, price, water used, soil moisture)
- Chart: Irrigation history (soil moisture + irrigation bars)
- Chart: Market trend with sentiment notes
- Graphic: Architecture (Data → ML → UI)
