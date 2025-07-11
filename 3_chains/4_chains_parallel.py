from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatGoogleGenerativeAI(
	model="gemini-2.5-flash",
	client_options=None,
	transport=None, 
	additional_headers=None,
	client=None,
	async_client=None
) # 替换为你想使用的 Gemini 模型名称

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)


# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features.",
            ),
        ]
    )
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features.",
            ),
        ]
    )
    return cons_template.format_prompt(features=features)


# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
def print_output(step_name, output):
    print(f"--- Output of {step_name} ---")
    print(output)
    print("-----------------------------")
    return output

chain = (
    prompt_template
    | RunnableLambda(lambda x: print_output("Prompt Template", x)) # 打印 prompt_template 输出
    | model
    | RunnableLambda(lambda x: print_output("Model", x))          # 打印 model 输出
    | StrOutputParser()
    | RunnableLambda(lambda x: print_output("StrOutputParser", x)) # 打印 StrOutputParser 输出
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: print_output("RunnableParallel", x)) # 打印 RunnableParallel 输出 (包含 pros 和 cons 的输出)
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
    | RunnableLambda(lambda x: print_output("Combine Pros Cons", x)) # 打印 combine_pros_cons 输出
)


# Run the chain
result = chain.invoke({"product_name": "MacBook Pro"})

# Output
print(result)
