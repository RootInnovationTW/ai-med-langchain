from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI

class ClinicalChat:
    def __init__(self):
        self.sessions = {}
        self.model = ChatOpenAI(temperature=0)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a clinical decision support assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        self.runnable = self.prompt_template | self.model
        self.chat = RunnableWithMessageHistory(
            self.runnable, self.get_session_history,
            input_messages_key="input", history_messages_key="history"
        )

    def get_session_history(self, session_id):
        return self.sessions.setdefault(session_id, ChatMessageHistory())

    def run(self, text, session_id):
        return self.chat.invoke({"input": text}, config={"configurable": {"session_id": session_id}})

if __name__ == "__main__":
    chat = ClinicalChat()
    print(chat.run("List side effects of aspirin.", "case001"))
    print(chat.run("Is it safe before surgery?", "case001"))
