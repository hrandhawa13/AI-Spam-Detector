#pip install pandas scikit-learn 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv("spam.csv")

# Remove duplicates and update category
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

msg = data['Message']
category = data['Category']

# Split 20% data into the test dataset
(msg_train, msg_test, category_train, category_test) = train_test_split(msg, category, test_size=0.2)

# Ignore common English words like 'a', 'I', etc.
cv = CountVectorizer(stop_words='english')

# Convert training data into numerical form
numerical = cv.fit_transform(msg_train)

# Create the model
model = MultinomialNB()

# Train the model
model.fit(numerical, category_train)

# evaluate the model
numerical_test = cv.transform(msg_test)

# Print the score
print('Score is', model.score(numerical_test, category_test)* 100)

# Predict function
def predict(message):
    input_message = cv.transform([message]).toarray()
    print('Predicting following message: ' + message)
    return 'AI thinks this msg is ' + model.predict(input_message)[0]


print(predict('Congrats, you won a lottery'))
print(predict('This is your doctor appointment'))
print(predict('Congrats on your graduation, love from grandma'))
print(predict('Email from your manager. Why isnt this work completed yet? '))
print(predict('Click on this link'))