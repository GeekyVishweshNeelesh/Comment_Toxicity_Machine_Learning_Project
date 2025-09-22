💬 Toxic Comment Classification & Moderation App 🚫🤖

A machine learning–powered tool to automatically detect different types of toxic content in user comments.
The app classifies comments into multiple toxicity categories to help online platforms moderate harmful language effectively.


🧩 Overview

✨ Goal: Detect and classify user comments into:

🧨 Toxic

⚠️ Severe Toxic

🤬 Obscene

🔪 Threat

😡 Insult

🚷 Identity Hate

💻 User Interface: Built with Streamlit where users can type a comment and instantly see predictions.
🌍 Use Cases: Social media 🐦, forums 💬, customer reviews 🛒 – anywhere harmful content needs monitoring.


🔧 Tech Stack

🐍 Python 3.x

📚 scikit-learn

🚀 XGBoost (final model)

✍️ TF-IDF Vectorizer

🎨 Streamlit (UI)

💾 joblib (model persistence)




📂 Dataset & Features

📊 Columns in train.csv:

Column Name	Description
🆔 id	Unique comment ID
💬 comment_text	Raw comment text
🧨 toxic	1 = toxic
⚠️ severe_toxic	1 = severe toxic
🤬 obscene	1 = obscene
🔪 threat	1 = threat
😡 insult	1 = insult
🚷 identity_hate	1 = identity-based hate
⚙️ Preprocessing: cleaning text, stopword removal, lowercasing, TF-IDF transformation.



🧠 Models Tried
🤖 Model	🔍 Characteristics
📈 Logistic Regression	Baseline, fast, interpretable
🌲 Random Forest	Handles non-linearities, slower
🏆 XGBoost	Best balance of precision & recall



📈 Performance

✅ Metrics: Precision, Recall, F1-score (per label + overall)

🔄 Hyperparameter tuning improved rare-class detection (threat, identity_hate)

🎯 Final choice: XGBoost + TF-IDF





🛠️ Example Comments
Comment	Prediction
“You are absolutely worthless and everyone hates you.”	Toxic, Insult
“I will find you and make you pay …”	Threat, Toxic
“I really appreciated your post; great depth and insight!”	Non-toxic







