#!/usr/bin/env python
# coding: utf-8

# In[33]:


#importing libraries
import pandas as pd
#Model
from sklearn.metrics.pairwise import cosine_similarity
#Deployment
import pickle
import streamlit as st


# In[34]:


# Function for popularity-based recommendations
def popularity_recommendation_with_titles(df, n_recommendations=5):
    # Group by ISBN and count the number of ratings
    book_popularity = df.groupby('ISBN')['book_rating'].count().reset_index(name='popularity')

    # Sort books by popularity in descending order
    sorted_books = book_popularity.sort_values(by='popularity', ascending=False)

    # Get the top N recommendations
    top_recommendations = sorted_books.head(n_recommendations)

    # Extract ISBNs from the recommendations
    recommendations_isbn = top_recommendations['ISBN'].tolist()

    # Extract book titles using the ISBN codes
    recommendations_titles = [
        df[df['ISBN'] == isbn]['booktiltle'].iloc[0] for isbn in recommendations_isbn
    ]

    return recommendations_titles


# In[35]:


# Read the full DataFrame
df = pd.read_csv('/Users/azadkabira/Desktop/Internship/Model/final/Combined Dataframes.csv')


# In[36]:


# Downsize the DataFrame using popularity-based recommendations
downsized_books = popularity_recommendation_with_titles(df, n_recommendations=100)  # Choose an appropriate number


# In[37]:


# Create a new DataFrame based on the downsized book list
new_df = df[df['booktiltle'].isin(downsized_books)]


# In[38]:


# Calculate cosine similarity on the new DataFrame
book_reader_matrix = pd.get_dummies(new_df[['bookauthor', 'Location']])
book_reader_similarity = cosine_similarity(book_reader_matrix)


# In[39]:


user_input_id = input("Enter the userID: ")


# In[40]:


if int(user_input_id) not in df['userID'].values:
    print('Invalid UserID')
else:
    print('UserID: ', user_input_id)

    # Retrieving the index of the user
    user_index = df[df['userID'] == int(user_input_id)].index[0]

    # Recommend books for a specific reader
    similar_books = book_reader_similarity[user_index].argsort()[::-1][1:]  # Remove [:-1] and add [1:] for correct slicing

    # Display recommended books
    recommended_books = df.iloc[similar_books]['booktiltle']
    #print(recommended_books)


# In[41]:


similar_readers_with_scores = list(enumerate(book_reader_similarity[user_index]))
similar_readers_with_scores_sorted = sorted(similar_readers_with_scores, key=lambda x: x[1], reverse=True)
similar_readers = similar_readers_with_scores_sorted[1:]  # Excluding the user itself

# If you want to recommend books based on the most similar reader
most_similar_reader_index = similar_readers[0][0]
recommended_books_indices = book_reader_similarity[most_similar_reader_index].argsort()[:-1][::-1]
recommended_books = df.iloc[recommended_books_indices]['booktiltle']

# output lib will recommend top 5 books
output = recommended_books.head(5)

# Display or use recommended_books for further processing
#print("Recommended books for user", user_input_id, "based on similar user preferences:\n", output)


# In[42]:


'''
if user_input_id:
    user_input_id = int(user_input_id)

    # Check if the entered user ID exists in the DataFrame
    if user_input_id not in new_df['userID'].values:
        st.write('Invalid UserID')
    else:
        st.write('UserID: ', user_input_id)

        # Retrieve the index of the user
        user_index = new_df[new_df['userID'] == user_input_id].index[0]

        # Use cosine similarity on the new DataFrame for recommendations
        similarity_scores = book_reader_similarity[user_index]

        # Sort and get top similar users (excluding the user itself)
        similar_users_indices = similarity_scores.argsort()[::-1][1:]

        # Recommend books based on most similar user
        most_similar_user_index = similar_users_indices[0]
        recommended_books_indices = book_reader_similarity[most_similar_user_index].argsort()[::-1]
        # Display recommended books
        recommended_books = df.iloc[similar_books]['booktiltle']
        print(recommended_books)
'''


# In[48]:


'''
file_path = '/Users/azadkabira/Desktop/book_recommendation_model.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(book_reader_similarity, file)

print(f"Book similarity data saved to {file_path}")
'''


# In[ ]:




