import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("--- Sentiment Analysis Model ---")

# 1. Create a dummy dataset of reviews
data = {
    'Review': [
        "I absolutely love this product, it is amazing!",
        "This is the worst thing I have ever bought.",
        "Fantastic experience, highly recommend.",
        "Terrible quality, very disappointed.",
        "Great customer service and fast shipping.",
        "Complete waste of money, do not buy."
    ],
    'Sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Negative']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# 2. Convert text to numerical data (Feature Extraction)
# We use CountVectorizer to count the frequency of words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

# 3. Train the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X, y)
print("Model trained successfully on sample data.\n")

# 4. Test the model with user input
print("--- Sentiment Analysis with User Input ---")
print("Type 'exit' to quit\n")

while True:
    user_review = input("Enter a review: ").strip()
    
    if user_review.lower() == 'exit':
        print("Thank you for using Sentiment Analysis. Goodbye!")
        break
    
    if not user_review:
        print("Please enter a valid review!\n")
        continue
    
    # Transform the new text using the SAME vectorizer
    X_user = vectorizer.transform([user_review])
    
    # Make prediction
    prediction = model.predict(X_user)[0]
    
    # Display the result
    print(f"Review: '{user_review}'")
    print(f"Predicted Sentiment: {prediction}\n")