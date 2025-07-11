# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# Google Gemini Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/google_generative_ai/

from dotenv import load_dotenv
# 导入 Google Gemini 的聊天模型类
from langchain_google_genai import ChatGoogleGenerativeAI
import os # 导入 os 库以便访问环境变量

# Load environment variables from .env
load_dotenv()

# Create a ChatGoogleGenerativeAI model
# 您需要将 'gemini-pro' 替换为您想使用的具体 Gemini 模型名称
# 可以通过 API KEY 参数或环境变量 GOOGLE_API_KEY 提供密钥
# 例如：model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
# 如果您的 GOOGLE_API_KEY 已经通过 dotenv 加载，下面的方式也可以
model = ChatGoogleGenerativeAI(
	model="gemini-2.5-flash",
	client_options=None,
	transport=None, 
	additional_headers=None,
	client=None,
	async_client=None
) # 替换为你想使用的 Gemini 模型名称

# Invoke the model with a message
result = model.invoke("你好啊")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
