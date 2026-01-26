import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from prompt import system_prompt

# 1. Initialize FastAPI
app = FastAPI()

# 2. Setup Templates and Static Files

app.mount("/static", StaticFiles(directory="static"), name="static") 
templates = Jinja2Templates(directory="templates")

# Load Environment Variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- LangChain Logic ( ---
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "input": RunnablePassthrough()}
    | prompt
    | chatModel
)
# -----------------------------------

# 3. Route for the Home Page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# 4. Route for the Chat Logic
@app.post("/get")
async def chat(msg: str = Form(...)):
    input_text = msg
    print(input_text)
    
    # Invoke the chain
    response = rag_chain.invoke(input_text)
    
    # Handle response extraction
    answer = response.content if hasattr(response, 'content') else str(response)
    print("Response : ", answer)
    
    return answer

if __name__ == '__main__':
    # 5. Run with Uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)