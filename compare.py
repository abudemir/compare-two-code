import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(file1, file2):
    # Read the contents of the files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        file1_content = f1.read()
        file2_content = f2.read()

    # Tokenize the contents into sentences
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    file1_tokens = tokenizer.tokenize(file1_content)
    file2_tokens = tokenizer.tokenize(file2_content)

    # Convert the tokens to lowercase
    file1_tokens = [token.lower() for token in file1_tokens]
    file2_tokens = [token.lower() for token in file2_tokens]

    # Convert the tokens back to string
    file1_text = ' '.join(file1_tokens)
    file2_text = ' '.join(file2_tokens)

    # Vectorize the documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([file1_text, file2_text])

    # Calculate the cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity_score

# Example usage
file1 = 'csv-to-opensearch.py'
file2 = 'plans-007a.py'
similarity = calculate_similarity(file1, file2)
print(f"The similarity between {file1} and {file2} is: {similarity}")

