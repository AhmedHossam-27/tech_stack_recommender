import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'Job_Role': [
        'DevOps Engineer', 
        'Data Scientist', 
        'Cloud Architect', 
        'Web Developer', 
        'AI Specialist'
    ],
    'Skills': [
        'Python Cloud Automation Linux Docker Jenkins',
        'Python Machine-Learning SQL Statistics R Data-Analysis',
        'Cloud AWS Azure Security Networking Automation Infrastructure',
        'HTML CSS JavaScript React Web-Design Frontend Backend',
        'Python Neural-Networks Deep-Learning TensorFlow Computer-Vision AI'
    ]
}

df = pd.DataFrame(data)

def get_recommendations(user_skills, df):
    """
    Calculates cosine similarity between user skills and job roles,
    returning the top 3 recommendations.
    """
    tfidf = TfidfVectorizer()
    
    all_descriptions = list(df['Skills']) + [user_skills]
    
    tfidf_matrix = tfidf.fit_transform(all_descriptions)
    
    jobs_vectors = tfidf_matrix[:-1]    
    user_vector = tfidf_matrix[-1]      
    
    similarity_scores = cosine_similarity(user_vector, jobs_vectors).flatten()
    
    df_result = df.copy()
    df_result['Similarity_Score'] = similarity_scores
    
    df_sorted = df_result.sort_values(by='Similarity_Score', ascending=False)
    
    top_3_recommendations = df_sorted.head(3)
    
    return top_3_recommendations

print("=== AI Career Recommendation System ===")
print("Please enter at least 3 skills you prefer (separated by spaces):")
print("Example: Python Cloud Automation\n")

user_input = input("Enter your skills: ")

if not user_input.strip():
    print("\n[Cold Start Warning]: No skills entered. Reverting to Trending Fallbacks:")
    for index, row in df.head(3).iterrows():
        print(f"- {row['Job_Role']}")
else:
    recommendations = get_recommendations(user_input, df)
    
    print("\n=== TOP 3 RECOMMENDED ROLES FOR YOU ===")
    for rank, (index, row) in enumerate(recommendations.iterrows(), 1):
        match_percentage = round(row['Similarity_Score'] * 100, 2)
        print(f"{rank}. {row['Job_Role']} (Match Score: {match_percentage}%)")
        print(f"   Required Skills: {row['Skills']}\n")