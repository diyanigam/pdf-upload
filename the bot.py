from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_phi"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PHI2_MODEL_NAME = "microsoft/phi-2" # This is the Phi-2 model identifier

# --- 1. Load your Embedding Model ---
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- 2. Load your Chroma Vector Database ---
# Ensure you reload the database so it's accessible
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 most relevant documents

# --- 3. Load the Phi-2 LLM ---
# Using BitsAndBytes for 4-bit quantization to make it super lightweight
# You'll need a GPU for this to work effectively, or adjust to cpu for local
quantization_config = {
    'load_in_4bit': True,
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_quant_type': "nf4",
    'bnb_4bit_compute_dtype': torch.bfloat16
}

tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    PHI2_MODEL_NAME,
    quantization_config=torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8) if torch.cuda.is_available() else None, # Apply quantization only if CUDA is available, otherwise use CPU
    device_map="auto",
    trust_remote_code=True
)

# Create a HuggingFace pipeline for text generation
# Phi-2 is a causal language model, so we'll use 'text-generation'
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256, # Limit the response length
    do_sample=True,
    top_k=50,
    temperature=0.7,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id # Set pad_token_id to eos_token_id for Phi-2
)

# Wrap the HuggingFace pipeline in a LangChain compatible LLM
# This is a basic wrapper, you might need a more sophisticated one for production
class CustomHuggingFaceLLM:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def invoke(self, prompt: str) -> str:
        result = self.pipeline(prompt)
        return result[0]['generated_text']

llm = CustomHuggingFaceLLM(llm_pipeline)

# --- 4. Define your Prompt Template ---
# This template instructs the LLM on how to use the retrieved context.
template = """
You are a helpful AI assistant specialized in providing information about the user you represent.
Use the following context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 5. Construct the RAG Chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | RunnableLambda(lambda x: llm.invoke(x.messages[0].content)) # Extract content for the custom LLM
    | StrOutputParser()
)

# --- 6. Chat Function ---
def chat_with_phi2(query: str):
    response = rag_chain.invoke(query)
    # The Phi-2 model might repeat the prompt or add conversational filler.
    # We'll try to clean it up. This might need fine-tuning.
    clean_response = response.split("Answer:")[
        -1
    ].strip()  # Try to extract content after "Answer:"
    return clean_response

# --- Example Usage ---
if __name__ == "__main__":
    print("Welcome to your personal RAG chatbot! Ask me anything about yourself.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        response = chat_with_phi2(user_query)
        print(f"Bot: {response}")