import azure.functions as func
import logging
import pandas as pd
import openai
import numpy as np
import pickle
import json
import random
import secrets
from transformers import GPT2TokenizerFast

COMPLETIONS_MODEL = "text-davinci-003"
openai.api_key = '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'


MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

with open('context_embeddings.pkl', 'rb') as handle:
        context_embeddings = pickle.load(handle)
        handle.close()

df = pd.read_csv('all_biz_all.csv')
df = df.set_index(["title", 'url'])

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

f"Context separator contains {separator_len} tokens"

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    response = {
        "prompt": prompt,
        "response": response["choices"][0]["text"].strip(" \n")
    }
    return response

def save_conversation(conversation: str, filename: str):
    with open(f'./conversation/{filename}.pkl', 'wb') as handle:
        pickle.dump(conversation, handle)

def get_conversation(filename: str):
    with open(f'./conversation/{filename}.pkl', 'rb') as handle:
        conversation = pickle.load(handle)
    return conversation

app = func.FunctionApp()

# Learn more at aka.ms/pythonprogrammingmodel

# Get started by running the following code to create a function using a HTTP trigger.

@app.function_name(name="HttpTrigger1")
@app.route(route="QA")
def main_(req: func.HttpRequest) -> func.HttpResponse:
    request_body = json.loads(req.get_body().decode('utf-8'))
    question = request_body['question']

    #check if request body contains conversation secret
    if 'conversation_secret' in request_body:
        conversation = get_conversation(request_body['conversation_secret'])
        question = conversation + question
    else:
        request_body['conversation_secret'] = secrets.token_hex(16)

    tmp = order_document_sections_by_query_similarity(question, context_embeddings)[:5]
    ans = answer_query_with_context(f'{question} answer in czech', df, context_embeddings)
    
    conversation = ans['prompt'] + ans['response']
    
    save_conversation(conversation, request_body['conversation_secret'])

    return_body = {
        "answer": ans['response'],
        "confidence": tmp[0][0],
        "article_title": tmp[0][1][0],
        "article_url": tmp[0][1][1],
        "conversation_secret": request_body['conversation_secret']

    }

    if ans:
        return func.HttpResponse(json.dumps(return_body), status_code=200, mimetype="application/json")
    else:
        return func.HttpResponse(
                "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
                status_code=200
        )