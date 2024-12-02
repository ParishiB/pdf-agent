
# add memory to it
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from dotenv import load_dotenv
load_dotenv()

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    add_chat_history_to_messages=True,
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    ),
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load()

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)
agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
