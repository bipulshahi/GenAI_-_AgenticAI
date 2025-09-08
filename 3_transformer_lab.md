pip install transformers

"""**Text classification using BERT**

- Bidirectional Encoder Representations from Transformers
"""

from transformers import pipeline

classifier = pipeline('sentiment-analysis')     #Encoder

#Define text
texts = ["I like transformers!" , "Sometimes machine models behaves terrible"]
results = classifier(texts)
print(results)

"""**2. Text generation**"""

from transformers import pipeline

translator = pipeline('text-generation' , model="gpt2")    #Decoder

output = translator("Roses are red and sky is blue" , max_length=2)
print(output[0]['generated_text'])

"""**3. Machine transalation**"""

from transformers import pipeline

translator = pipeline("translation" , model="Helsinki-NLP/opus-mt-en-hi")    #Encoder - Decoder

output = translator("Roses are red and sky is blue")
print(output[0]['translation_text'])

"""**4. Text Summarization** - google-t5/t5-base"""

from transformers import pipeline

summary = pipeline("summarization" , model="google-t5/t5-base" ,
                   tokenizer="google-t5/t5-base")     #Encoder - Decoder

text = """
BERT, which stands for Bidirectional Encoder Representations from Transformers,
is a powerful language model developed by Google in 2018. It's designed to improve how computers
understand and process human language by considering the context of words in a sentence, both before and after them.
This bidirectional approach, combined with the transformer architecture, allows BERT to achieve state-of-the-art results on various NLP tasks
"""

output = summary(text, max_length = 50, min_length = 10 , do_sample=False)

print("Summary-" , output[0]['summary_text'])

"""**facebook/bart-large-cnn**"""

# Use a pipeline as a high-level helper
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
BERT, which stands for Bidirectional Encoder Representations from Transformers,
is a powerful language model developed by Google in 2018. It's designed to improve how computers
understand and process human language by considering the context of words in a sentence, both before and after them.
This bidirectional approach, combined with the transformer architecture, allows BERT to achieve state-of-the-art results on various NLP tasks
"""

output = summarizer(text, max_length = 50, min_length = 10 , do_sample=False)

print(output[0]['summary_text'])
