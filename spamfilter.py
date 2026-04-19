import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline 
 
# Example training dataset 
data = { 
    "text": [ 
        "Congratulations you won a free lottery ticket", 
        "Claim your free prize now", 
        "Win money instantly click this link", 
        "Hi how are you doing today", 
        "Let's meet for lunch tomorrow", 
        "Can you send me the report", 
        "Limited time offer buy now", 
        "Are we still meeting today" 
    ], 
    "label": [1,1,1,0,0,0,1,0]  # 1 = spam, 0 = not spam 
} 
 
df = pd.DataFrame(data) 
 
# Build ML pipeline 
model = Pipeline([ 
    ("vectorizer", TfidfVectorizer()), 
    ("classifier", MultinomialNB()) 
]) 
 
# Train the model 
model.fit(df["text"], df["label"]) 
 
# -------- User Input Paragraph -------- 
paragraph = input("Enter a paragraph: ") 
 
# Prediction 
prediction = model.predict([paragraph])[0] 
 
if prediction == 1: 
    print("Result: SPAM") 
else: 
    print("Result: NOT SPAM")