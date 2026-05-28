import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

df = pd.read_csv("maildata.csv")

print(df.shape)

df.drop_duplicates(inplace=True)
df['Category'] = df['Category'].replace(['ham','spam'],['Not Spam','Spam'])
print(df.head())

mess = df['Message']
cat = df['Category'] 

mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train) 


# aditya ka model 

model = MultinomialNB()
model.fit(features, cat_train)

# testing the  aditya ka model
test_features = cv.transform(mess_test)

# score prediction ke liye
#predictions = model.score(test_features, cat_test)
#print(predictions)

# predict karne ke liye
def predict(message):
    message_features = cv.transform([message])
    prediction = model.predict(message_features)
    return prediction

st.header("Email Spam Detection")


#output =predict("Congratulations! You've won a free ticket to the Bahamas! Click here to claim your prize.")

#print(output)
# test karne ke liye
#message = ["Congratulations! You've won a free ticket to the Bahamas! Click here to claim your prize."] 
#message_features = cv.transform(message.toarray())
#predictions = model.predict(message_features)
#print(predictions)

input_mess = st.text_input("Enter your email message here:")
if st.button("Predict"):
    result = predict(input_mess)
    st.write(f"The email is classified as: {result[0]}")





