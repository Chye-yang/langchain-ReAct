import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain # 注释掉旧版导入
# from langchain.chains.combine_documents import create_stuff_documents_chain # 注释掉旧版导入
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__name__)) # 使用 __file__ 获取当前文件路径更准确
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm  = ChatGoogleGenerativeAI(
	model="gemini-2.5-flash",
	client_options=None,
	transport=None,
	additional_headers=None,
	client=None,
	async_client=None
) # 替换为你想使用的 Gemini 模型名称


# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever using LCEL
# This uses the LLM to help reformulate the question based on chat history
# 旧版 history_aware_retriever = create_history_aware_retriever( llm, retriever, contextualize_q_prompt ) # 注释掉旧版创建

# LCEL 版本 History Aware Retriever
history_aware_retriever = contextualize_q_prompt | llm | StrOutputParser() | retriever

history_aware_retriever

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering using LCEL
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
# 旧版 question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) # 注释掉旧版创建

# LCEL 版本 Document Combination Chain
document_chain = qa_prompt | llm | StrOutputParser()


# Create a retrieval chain that combines the history-aware retriever and the question answering chain using LCEL
# 旧版 rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) # 注释掉旧版创建

# LCEL 版本 RAG 链
rag_chain = (
    RunnableParallel(
        context=history_aware_retriever,
        input=RunnablePassthrough() # Pass the input to the document chain
    ) | document_chain
)


# Function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        # LCEL invoke 需要一个字典作为输入，包含 input 和 chat_history
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result}") # LCEL Chain 的输出通常直接是结果
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        # 将 AI 的回复改为 AIMessage
        chat_history.append(AIMessage(content=result))



# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()
