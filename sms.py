import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and preprocess the data
sms_df = pd.read_csv(r'C:\Users\DELL\Desktop\SMSSpamCollection.csv')
sms_df.rename(columns={sms_df.columns[0]:'Label',sms_df.columns[1]:'Text'},inplace=True)
sms_df.drop_duplicates(inplace=True)
sms_df['Label'] = sms_df['Label'].replace({'ham':0,'spam':1})

# Train the model
cv = CountVectorizer(stop_words='english', max_features=10000)
x_vec = cv.fit_transform(sms_df['Text']).toarray()
y = sms_df['Label'].values

ml = MultinomialNB()
ml.fit(x_vec, y)

# Define the Streamlit app
def main():
    st.title("SMS Spam Classification")
    st.subheader("Check if a message is ham or spam")

    # Create an input box for the user to enter a message
    message = st.text_input("Enter a message")

    # Create a button to trigger the prediction
    if st.button("Predict"):
        # Preprocess the user's input message
        processed_message = cv.transform([message]).toarray()

        # Make a prediction using the trained model
        prediction = ml.predict(processed_message)

        # Display the prediction result
        if prediction[0] == 0:
            st.write("The message is classified as 'ham'.")
        else:
            st.write("The message is classified as 'spam'.")

if __name__ == '__main__':
    main()
