from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
llm=ChatGoogleGenerativeAI(
model="gemini-2.5-flash-lite",
temperature=0.7,
)
response=llm.invoke("What is the population of India?")
print(response.content)