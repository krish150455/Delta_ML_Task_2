from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (faithfulness,answer_relevancy,context_precision, context_recall)

#PDF INGESTION + ADDING EXTRA METADATA

#IMPROVING METADATA, BY ADDING EXTRA METADATA:
EXTRA_METADATA = {
    "annual-report-2025-national-recovery-plan-koala.pdf": {
        "report_years": "2025",
        "report_type": "national_recovery_plan"
    },
    "koala-conservation-inffer-report-2022.pdf": {
        "report_years": "2022-2023",
        "report_type": "conservation_inffer_report"
    },
    "koala-strategy-first-implementation-report.pdf": {
        "report_years": "2009-2014",
        "report_type": "first_implementation_report"
    }
}

#EXTRACTING DATA FROM ALL THE PAGES OF ALL DOCUMENTS AND ADD TO A SINGLE LIST - PDF DOCUMENT INGESTION
converter = DocumentConverter()

os.makedirs("markdown", exist_ok=True)

files = os.listdir("pdfs")
all_docs = []

for file in files:
    pdf_path = os.path.join("pdfs", file)

    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()
    md_path = os.path.join(
        "markdown",
        file.replace(".pdf", ".md")
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    loader = TextLoader(md_path, encoding="utf-8")
    documents = loader.load()

    metadata = EXTRA_METADATA[file]

    for doc in documents:
        doc.metadata["source"] = file
        doc.metadata["report_years"] = metadata["report_years"]
        doc.metadata["report_type"] = metadata["report_type"]

    all_docs.extend(documents)


#CHUNKING

#NEXT WE WILL SPLIT INTO CHUNKS RECURSIVELY, THAT IS, PARAGRAPHS -> SENTENCES -> WORDS AND SO ON ALL
#MAINTAINING SEMANTIC MEANING.
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Heading1"),
        ("##", "Heading2"),
        ("###", "Heading3")
    ]
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
#DOING SOME CHUNKING EXPERIMENTS HERE
#(1000,200)-pretty bad with lots of duplicates, even disclaimer page/index came - diluted embeddings - better context
#(500,100)- better, showing small chunks mean more focused, better and focused embeddings - better precision
#(800,100) - better than first one with lower duplication but kinda diluted response again

chunks = []

for doc in all_docs:
    header_chunks = header_splitter.split_text(doc.page_content)

    for chunk in header_chunks:
        chunk.metadata.update(doc.metadata)
        final_chunks = recursive_splitter.split_documents( [chunk] )
        chunks.extend(final_chunks)

#USING CHROMA DB AS VECTOR DB

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="./chroma_db")
#we supply the chunks to store, tell the model to embedd the chunks into vectors using that model
#and finally, a location to persistently save all vectors, metadata, docs, indexes to disk.

#CURATING THE BEST RETRIEVAL
query = input("Enter Query:")

#ENHANCING THE METADATA ADJUSTMENTS BY USING FILTER
def create_filter(query):
    filter_dict = {}
    if "2025" in query:
        filter_dict["report_type"] = "national_recovery_plan"
    elif "2022" in query or "2023" in query:
        filter_dict["report_type"] = "conservation_inffer_report"
    elif "2010" in query or "2011" in query:
        filter_dict["report_type"] = "first_implementation_report"
    return filter_dict

#BETTER: Using MMR
#As above commented code, currently we are retrieving top-k searches based on similarity.
#But, these might have duplicates.
#Instead, we use MMR method-Maximum Marginal Relevance method which retrieves more and selects "disinct and relevant"
#ones from them.

def create_retriever(query):
    filter_dict = create_filter(query)
    if filter_dict:
        return vectordb.as_retriever(search_type="mmr",search_kwargs={"k": 12,"fetch_k": 30,"filter": filter_dict})
    return vectordb.as_retriever(search_type="mmr",search_kwargs={"k": 12,"fetch_k": 30})

#QUERY DECOMPOSITION
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))
local_llm = ChatOllama(model="phi3")
def decompose_query(query):
    decompose_prompt = f"""
    Break this question into retrieval questions.
    Question:
    {query}
    Return one query per line.
    """
    response = local_llm.invoke(decompose_prompt)
    queries = [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]
    return queries

#MULTIQUERY RETRIEVAL
#Semantic search heavily depends on the words in query. So instead of using just one query, retrieve
#with multiple semantically equivalent queries...so that even if the same word is not there, correct info is retrieved
def create_multi_queries(query):
    prompt = f"""
    Generate 5 alternative search queries for:
    {query}
    Return one query per line.
    """
    response = local_llm.invoke(prompt)
    queries = [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]
    return queries

#RETRIEVING ALL RELEVANT MATCHES
def retrieve_matches(query):
    retriever = create_retriever(query)
    matches = []
    subqueries = decompose_query(query)
    if len(subqueries) == 0:
        subqueries = [query]
    for subquery in subqueries:
        queries = create_multi_queries(subquery)
        for q in queries:
            docs = retriever.invoke(q)
            matches.extend(docs)
    unique = {}
    for doc in matches:
        key = (doc.metadata["source"],doc.page_content)
        unique[key] = doc

    return list(unique.values())

results = retrieve_matches(query)

#PROMPT GENERATION WITH TEMPLATE
context=""
for doc in results:
    context += f"""Source: {doc.metadata['source']} 
                Type: {doc.metadata['report_type']} Content: {doc.page_content}"""

prompt = PromptTemplate.from_template(
    """
Use the following context to answer the question.
If the answer is not present in the context, say:
"I don't know based on the provided documents."
When making a claim, cite:
- Source document
- Page number

If comparing reports, clearly separate findings from each report.
Provide a detailed answer.

Context:
{context}

Question:
{question}

Answer:
"""
)
#final_prompt = prompt.invoke({"context": context,"question": query})

#response = llm.invoke(final_prompt.text)

#print(response.content)


#RAGAS EVALUATION
def run_rag(query):
    results = retrieve_matches(query)
    contexts = [
        doc.page_content
        for doc in results
    ]
    context = ""

    for doc in results:
        context += f"""
Source: {doc.metadata['source']}

Type: {doc.metadata['report_type']}

Content:
{doc.page_content}

"""
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final_prompt.text)

    return response.content, contexts



#EVALUATION OF PIPELINE - THROUGH DEEPEVAL
evaluation_queries=["Based on the jurisdiction summaries in the 2010 report, how did the narrative description of conservation concerns in Victoria differ from those in New South Wales regarding primary threats?",
                    "According to the narrative findings of the INFFER report, what are the recommended strategies for prioritizing investment in New South Wales and Queensland?",
                    "Based on the 2025 Annual Report narrative, how is the National Recovery Plan coordinating efforts to address threats like disease and vehicle strikes?",
                    "How does the narrative of the 2025 Annual Report describe the change in governance and coordination compared to the jurisdiction-led structure described in the 2010 report?",
                    "How does the narrative text of the INFFER report describe the socio-economic and technical challenges faced by koala habitat restoration projects?"]


def evaluate_ragas(query, answer, contexts):

    dataset = Dataset.from_dict(
        {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "reference": [answer]
        }
    )

    result = evaluate(dataset=dataset,metrics=[faithfulness,answer_relevancy,context_precision,context_recall])

    print(result)

for query in evaluation_queries:
    print("QUESTION:")
    print(query)
    answer, context = run_rag(query)
    print("\nANSWER:\n")
    print(answer)
    print("\nRAGAS EVALUATION\n")
    evaluate_ragas(query,answer,context)