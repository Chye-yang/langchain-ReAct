# 部分 1: 导入所需的库
# ----------------------------------------------------------------------------
from dotenv import load_dotenv            # 用于从 .env 文件加载环境变量
from langchain import hub                 # 用于从 Langchain Hub 获取预定义资源
from langchain.agents import AgentExecutor, create_structured_chat_agent # 用于创建和执行代理
from langchain.memory import ConversationBufferMemory # 用于存储和管理对话历史
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage # 用于定义不同类型的聊天消息
from langchain_core.tools import Tool     # 用于定义代理可以使用的工具

# 导入用于 Google Generative AI 的 Langchain 集成
from langchain_google_genai import ChatGoogleGenerativeAI

# 部分 2: 配置和初始化
# ----------------------------------------------------------------------------
# 从 .env 文件加载环境变量（例如 API 密钥）
load_dotenv()

# 部分 3: 定义代理可以使用的工具
# ----------------------------------------------------------------------------

def get_current_time(*args, **kwargs):
    """
    工具函数：返回当前时间。
    返回:
        str: 当前时间，格式为 YYYY-MM-DD H:MM AM/PM。
    """
    import datetime
    now = datetime.datetime.now()
    # 修改strftime的格式，加入年、月、日
    return now.strftime("%Y-%m-%d %I:%M %p")


def search_wikipedia(query: str) -> str:
    """
    工具函数：搜索维基百科并返回第一个结果的摘要。

    Args:
        query (str): 要搜索的关键词。

    返回:
        str: 维基百科摘要，或在找不到信息时的提示信息。
    """
    from wikipedia import summary
    try:
        # 限制摘要长度为两句话以保持简洁
        return summary(query, sentences=2)
    except:
        return "未能找到相关信息。"

# 将工具函数封装成 Langchain 的 Tool 对象
tools = [
    Tool(
        name="Time",                # 工具名称
        func=get_current_time,      # 对应的函数
        description="需要知道当前时间时很有用。", # 工具的描述，用于代理决定何时使用该工具
    ),
    Tool(
        name="Wikipedia",           # 工具名称
        func=search_wikipedia,      # 对应的函数
        description="需要了解某个主题的信息时很有用。", # 工具描述
    ),
]

# 部分 4: 创建代理
# ----------------------------------------------------------------------------
# 从 Langchain Hub 拉取预定义的结构化聊天代理提示
# 提示定义了代理如何处理用户输入和工具
prompt = hub.pull("hwchase17/structured-chat-agent")

# 初始化 ChatGoogleGenerativeAI 模型
# 选择合适的 Gemini 模型名称
# 根据 Gemini 文档或 Langchain 文档，System Instructions 可以通过模型参数传递，
# 或者通过 Prompt 模板中的特定变量传递。
# 这里我们依赖于 structured-chat-agent 模板如何处理 System 类型的消息或指令。
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    # 其他可选参数保持不变
    client_options=None,
    transport=None,
    additional_headers=None,
    client=None,
    async_client=None
)

# 初始化对话内存
# ConversationBufferMemory 存储所有消息， allowing the agent to recall previous turns
# IMPORTANT: We are NOT adding the SystemMessage directly to the memory here.
# The system instruction will be handled by the prompt template and AgentExecutor.
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Memory key must match the variable name in the prompt template
    return_messages=True        # Ensures memory returns message objects
)

# 使用模型、工具和提示创建结构化聊天代理
# create_structured_chat_agent 是一个便捷函数，用于设置标准的聊天代理
agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)

# 部分 5: 设置代理执行环境
# ----------------------------------------------------------------------------
# AgentExecutor 是代理的核心执行器
# 它处理输入、将输入发送给代理、处理代理的输出（包括使用工具）以及管理记忆
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,              # 设置 verbose=True 会打印代理的思维过程，有助于调试
    memory=memory,             # 将内存附加到执行器，使其在运行时维护对话历史
    handle_parsing_errors=True, # 启用错误处理，以免在解析代理输出时中断
)

# 部分 6: 设置初始系统指令 (通过 prompt 传递)
# ----------------------------------------------------------------------------
# 定义系统指令的文本
# Instead of adding to memory, we rely on the prompt template to include this
# The structured-chat-agent prompt typically expects `input`, `chat_history`,
# and sometimes a specific variable for system or context instructions.
# We will pass this as part of the input dictionary to agent_executor.invoke
system_instruction_text = "你是一个人工智能助手，可以使用可用的工具提供有用的答案。\n如果你无法回答，你可以使用以下工具：Time 和 Wikipedia。"


# 部分 7: 聊天循环
# ----------------------------------------------------------------------------
print("AI助手已启动。输入 'exit' 退出。")

# 开始一个持续的聊天循环，与用户交互
while True:
    # 获取用户的输入
    user_input = input("用户: ")

    # 检查用户是否输入 'exit' 来结束对话
    if user_input.lower() == "exit":
        print("AI助手已退出。")
        break

    # 将用户的消息添加到对话内存中
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # 调用 AgentExecutor 处理用户输入并生成回复
    # 结构化代理的 Prompt 通常会有一个地方来接收系统指令。
    # AgentExecutor 会将这里的 input 字典传递给代理，代理会根据 Prompt 模板
    # 将这些值（包括 input 和通过 memory 提供的 chat_history）格式化后发送给模型。
    # 有些 structured-chat-agent 模板可能需要不同的输入 key 来接收系统指令，
    # 但标准的通常依赖于 Prompt 模板中的 'system_message', 'instructions', 等变量。
    # 默认情况下，AgentExecutor 会将 memory key ('chat_history') 和 input key ('input') 传递。
    # 具体的系统指令传递方式取决于 Agent 和 Prompt 的实现。
    # 对于许多结构化代理，将系统指令放入 Prompt 本身是更常见的方法，但如果你想动态设置，
    # 通常需要在 invoke 的输入字典中包含 Prompt 期望用于系统指令的 key。
    # 让我们尝试将系统指令文本也传递过去。查看 structured-chat-agent 提示模板的源码可以确认
    # 它是否接受一个用于系统指令的特定 key。如果没有，最直接的方法是修改 prompt 模板
    # 或者依赖于模型本身对首个 SystemMessage 的处理。
    # 由于我们移除了手动添加 SystemMessage，这里依赖于 Prompt 模板或 Gemini 模型
    # 如何处理聊天历史中的第一个消息（在这种情况下，第一个将是 HumanMessage）。

    # 更加保险的做法是，确保 prompt 模板包含 {system_instructions} 或类似变量，
    # 然后修改 invoke 调用以包含这个 key。
    # 但是对于标准的 hub.pull("hwchase17/structured-chat-agent") 提示，
    # 它通常期望的是将历史记录和当前输入结合起来。
    # 考虑到之前的错误是由于 memory 中出现了不正确的 SystemMessage，
    # 移除 SystemMessage from memory 是解决问题的核心。

    try:
        # AgentExecutor automatically includes memory['chat_history']
        # and the provided 'input' in the context passed to the agent's plan method,
        # which then uses the prompt template.
        response = agent_executor.invoke({
            "input": user_input,
            # 注意：这里的 "system_instructions" key 是一个假设！
            # 你需要查看 'hwchase17/structured-chat-agent' Prompt 模板的定义，
            # 确认它是否期望一个名为 "system_instructions" 或其他名称的变量来接收系统指令。
            # 如果模板中没有这样的变量，传递这个 key 可能无效或导致其他错误。
            # 如果模板不直接支持独立的系统指令变量，最好的方法是将指令硬编码到你自己的 Prompt 中，
            # 或者使用支持通过模型参数传递系统指令的模型和相应的 Langchain 集成。
            # 为了与移除内存中的 SystemMessage 的目标一致，我们暂时不传递这个额外的 key。
            # 最标准的 structured-chat-agent 模板可能仅依赖于它自身结构和聊天历史。
            # "system_instructions": system_instruction_text # <-- cautious approach
        })

        # 打印代理生成的回复
        print("机器人:", response["output"])

        # 将代理的回复添加到对话内存中
        # AgentExecutor 应该会处理工具的使用和结果，并将最终回复作为 AIMessage 加入内存。
        # 这里我们再次手动添加最终的 AIMessage，以确保内存完整。
        memory.chat_memory.add_message(AIMessage(content=response["output"]))

    except Exception as e:
        # 处理任何可能发生的错误，例如 API 调用失败或其他运行时错误
        print(f"处理请求时发生错误: {e}")
        print("机器人: 抱歉，我暂时无法处理你的请求。")
        # 保留用户输入在内存中，以便下次尝试时代理能看到它
        # 如果移除，会让对话失去连贯性。
        # 如果你希望在出错后回滚用户消息，可以取消下面代码的注释，但要小心处理。
        # print("Attempting to roll back last user message...")
        # if memory.chat_memory.messages and isinstance(memory.chat_memory.messages[-1], HumanMessage) and memory.chat_memory.messages[-1].content == user_input:
        #      memory.chat_memory.messages.pop() # 移除最后一条用户消息
        # else:
        #     print("Could not find last user message to roll back.")

