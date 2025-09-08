# Semantic Search and Question Answering with Sentence Transformers

This document provides a detailed explanation of a semantic search workflow designed for a chatbot, which retrieves the most relevant section from a document based on the meaning of a user's query. The workflow is implemented using Python's `PyPDF2`, `re`, and `sentence-transformers` libraries, as shown in the `genai_g2_d11.py` script.

## 1. Semantic Search Workflow Overview

Semantic search is an advanced search technique that uses the meaning or intent of a query to find relevant information, rather than just matching keywords. This workflow demonstrates how to build a semantic search system for a question-answering chatbot, following these steps:

1.  **Install Required Dependencies:** `sentence-transformers` and `PyPDF2`.
2.  **Extract Text Data:** Read text from a PDF file.
3.  **Clean and Preprocess Text:** Remove unwanted characters and extra spaces.
4.  **Load a Pre-trained Model:** Use a Sentence Transformer model to convert text to embeddings.
5.  **Segment Document:** Break down the document text into logical sections.
6.  **Create Embeddings:** Generate vector representations for each document section and for the user's query.
7.  **Calculate Similarity:** Compute the cosine similarity between the query embedding and each section embedding.
8.  **Identify Best Match:** Find the section with the highest similarity score.

## 2. Step-by-Step Implementation

### 2.1. Installation and Text Extraction

* **Dependencies:** The script first installs `sentence-transformers` for creating semantic embeddings and `PyPDF2` for handling PDF files.
    ```bash
    !pip install sentence-transformers
    !pip install PyPDF2
    ```
* **PDF Extraction:** A function `extract_data_from_pdf` is defined to read a PDF file in binary mode (`'rb'`), iterate through all its pages using `PyPDF2.PdfReader`, and concatenate the text from each page into a single string.
    ```python
    import PyPDF2
    def extract_data_from_pdf(pdf_path):
        with open(pdf_path , 'rb') as file:
            pdfreader = PyPDF2.PdfReader(file)
            full_text = ''
            for page in pdfreader.pages:
                full_text += page.extract_text()
        return full_text
    extracted_text = extract_data_from_pdf('/content/company_manual.pdf')
    ```

### 2.2. Text Preprocessing

* **Cleaning Function:** A `clean_text` function is defined using the `re` (regular expression) module. It performs two key cleaning operations:
    1.  Removes extra whitespace (`\s+`) by replacing it with a single space.
    2.  Removes any non-ASCII characters (`[^\x00-\x7F]+`).
    ```python
    import re
    def clean_text(text):
        text = re.sub(r'\s+' , ' ' , text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text
    cleaned_text = clean_text(extracted_text)
    ```

### 2.3. Embedding Model and Document Segmentation

* **Sentence Transformer Model:** The `sentence-transformers` library provides access to pre-trained models. The script loads the `all-MiniLM-L6-v2` model, which is a lightweight but powerful model capable of generating **384-dimensional semantic embeddings** for sentences or paragraphs.
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ```
* **Document Segmentation:** The cleaned text is manually segmented into logical sections based on keywords like "About the Company", "Return Policy", etc. The resulting sections are stored in a dictionary, with section titles as keys and their corresponding text content as values.
    ```python
    sections = {
        "About the Company": cleaned_text.split('About the Company')[1].split('Return Policy')[0],
        "Return Policy": cleaned_text.split('Return Policy')[1].split('Warranty')[0],
        # ... and so on for other sections
    }
    ```

### 2.4. Embedding Generation and Cosine Similarity

* **Embedding Generation:**
    * The `model.encode()` method is used to generate a semantic embedding (vector representation) for the user's query.
    * The same method is used in a loop to generate and store embeddings for each of the document sections in a `section_embeddings` dictionary.
    ```python
    query = "How do I send the product back?"
    query_embedding = model.encode([query])[0]
    section_embeddings = {}
    for title,content in sections.items():
      section_embeddings[title] = model.encode([content])[0]
    ```
* **Cosine Similarity:**
    **Definition:** Cosine similarity is a metric that measures the cosine of the angle between two non-zero vectors. A score of 1 indicates the vectors are in the same direction (perfect similarity), 0 indicates they are orthogonal (no similarity), and -1 indicates they are in opposite directions. It is a standard method for quantifying how semantically similar a query is to a document section.
    * A function `cosine_similarity` is defined using `numpy`'s dot product and vector norms (`np.linalg.norm`).
    * The function is then used to calculate the similarity score between the `query_embedding` and each of the `section_embeddings`. The scores are stored in a `similarities` dictionary.
    ```python
    def cosine_similarity(a,b):
      return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))
    similarities = {}
    for title,emb in section_embeddings.items():
      similarity_score = cosine_similarity(query_embedding , emb)
      similarities[title] = similarity_score
    print(similarities)
    ```

### 2.5. Visualization and Chatbot Function

* **Visualization:** A horizontal bar plot (`matplotlib.pyplot.barh`) is used to visualize the calculated similarity scores, making it easy to see which section has the highest score.
    ```python
    import matplotlib.pyplot as plt
    plt.barh(list(similarities.keys()) , list(similarities.values()))
    plt.show()
    ```
* **Chatbot Function:** A `semantic_search` function encapsulates the entire workflow. It takes a `query`, the `sections` dictionary, and the `model` as input. It finds the section with the highest similarity score and returns a formatted response containing the title of the most relevant section and its content.
    ```python
    def semantic_search(query , sections , model):
      query_embedding = model.encode([query])[0]
      default_similarity = 0.0
      best_match = None
      for title,content in sections.items():
        section_embedding = model.encode([content])[0]
        similarity_score = cosine_similarity(query_embedding , section_embedding)
        if similarity_score > default_similarity:
          default_similarity = similarity_score
          best_match = title
      if best_match:
        return f"Bot: The most relevant section is {best_match}\nHere is the information:\n{sections[best_match]}"
      else:
        return "Bot: I couldn't find the relevant answer."
    ```
* **Example Usage:**
    The function is called with a query related to the return policy, and it correctly identifies "Return Policy" as the most relevant section.
    ```python
    response = semantic_search("send the product back" , sections , model)
    print(response) # Output: The bot response with the Return Policy section
    ```

## 3. Key Concepts, Benefits, and Limitations

* **Key Concepts:** This script illustrates **Semantic Search**, the use of **Sentence Transformers** to create semantic embeddings, and the application of **Cosine Similarity** to find the most relevant information based on meaning rather than keywords.
* **Benefits:** This approach provides a robust and intelligent way to answer questions, as it can understand the intent behind a query even if the exact keywords are not present in the document.
* **Limitations:** The effectiveness of this system depends on the quality of the pre-trained embedding model and the logical segmentation of the document. For very long documents or ambiguous queries, the manual segmentation and simple similarity score might not be sufficient to provide a perfectly accurate answer.
