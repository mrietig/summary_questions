import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from collections import Counter
import re

from openai import OpenAI

client_name = 'ekster'

client = OpenAI(
    api_key="credentials",
)

# Load the CSV file
question_df = pd.read_csv('Summary Questions/summary_question_set_{}.csv'.format(client_name))

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Preprocess the text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

question_df['processed_question'] = question_df['summarize_question'].apply(preprocess_text)

# Use BERT to embed the questions
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(question_df['processed_question'].tolist())

# Determine the optimal number of clusters using Silhouette score
def determine_optimal_clusters(data, min_k=2, max_k=10):
    iters = range(min_k, max_k + 1)
    s_scores = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        s_scores.append(silhouette_score(data, kmeans.labels_))
    
    optimal_k = iters[s_scores.index(max(s_scores))]
    return optimal_k, s_scores

# Set the minimum and maximum number of clusters to explore
min_clusters = 5
max_clusters = 20

optimal_clusters, s_scores = determine_optimal_clusters(embeddings, min_clusters, max_clusters)
print(f"The optimal number of clusters is: {optimal_clusters}")

# optimal_clusters = 15  # Uncomment and set this value to override the automatic selection

# Cluster the questions using K-Means with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
question_df['cluster'] = kmeans.fit_predict(embeddings)

# Function to generate a summary question for each cluster using GPT-4o
def generate_summary_question(cluster_num):
    cluster_questions = question_df[question_df['cluster'] == cluster_num]['processed_question']
    combined_questions = ' '.join(cluster_questions.tolist())
    
    max_tokens = 2048 
    if len(combined_questions) > max_tokens:
        combined_questions = combined_questions[:max_tokens]
    
    prompt = f"The following keywords are from a cluster of support agent tickets. Generate a single, short (not more than 15 words), general summary question based on them that encompasses issues the tickets might address: {combined_questions}"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates summary questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.7
    )
    
    generated_text = response.choices[0].message.content.strip()
    return generated_text

# Gather all cluster info
clusters_info = []
for cluster_num in range(optimal_clusters):
    count = len(question_df[question_df['cluster'] == cluster_num])
    automation_rate = question_df[question_df['cluster'] == cluster_num]['automated'].mean() * 100
    summary_question = generate_summary_question(cluster_num)  # One LLM call per cluster
    clusters_info.append({
        'cluster_num': cluster_num,
        'count': count,
        'summary_question': summary_question,
        'automation_rate': automation_rate
    })

# Sort clusters by count in descending order
clusters_info = sorted(clusters_info, key=lambda x: x['count'], reverse=True)

# Create a dictionary for cluster summaries
cluster_summary_dict = {info['cluster_num']: info['summary_question'] for info in clusters_info}

# Add a summary question column to the dataframe
question_df['summary_question'] = question_df['cluster'].map(cluster_summary_dict)

# Display the dataframe with the added summary question column
print(question_df[['summarize_question', 'cluster', 'summary_question', 'automated']])

# Display the sorted cluster information
for info in clusters_info:
    print(f"Cluster {info['cluster_num']}:")
    print(f"  Number of questions: {info['count']}")
    print(f"  Summary Question: {info['summary_question']}")
    print(f"  Automation Rate: {info['automation_rate']:.2f}%")
    print("\n")

# Save the dataframe to a CSV file
question_df.to_csv('Summary Questions/summary_question_set_categorized_{}.csv'.format(client_name), index=False)
