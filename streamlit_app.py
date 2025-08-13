import streamlit as st
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)

st.title("学习助手")

with st.sidebar:
    subject = st.selectbox(
        "选择学科领域",
        options=[
            "文学",
            "数学",
            "计算机",
        ],
        index=0,
    )

    style = st.selectbox(
        "讲解风格",
        options=["简洁", "详细"],
        index=0,
    )


def get_prompt_template(subject, style):

    style_dict = {
        "简洁": "仅提供直接答案和最少的必要解释。不要添加额外细节、发散讨论或无关信息。保持回答清晰、简洁，目标是为用户快速提供解决方案。",
        "详细": "第一，针对用户提问给出直接答案和清晰的解释；第二，基于此提供必要的相关知识点的信息，以补充背景或加深理解。",
    }

    system_prompt_template = "你是{subject}领域的专家，请你回答用户提问。你应当礼貌拒绝与该学科无关的问题。你需要遵循以下讲解风格：{style}。"
    system_prompt = SystemMessagePromptTemplate.from_template(
        template=system_prompt_template,
        partial_variables={"subject": subject, "style": style_dict[style]},
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            MessagesPlaceholder(variable_name="history"),
            human_prompt,
        ]
    )
    return prompt_template


prompt_template = get_prompt_template(subject, style)

client = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=st.secrets["OPENAI_API_KEY"],
)


def generate_response(user_input, prompt_template, memory, llm=client):
    chain = ConversationChain(llm=llm, memory=memory, prompt=prompt_template)
    response = chain.invoke({"input": user_input})
    return response["response"]


if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryMemory(
        return_messages=True, llm=client
    )
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好，我是你的学习助手！"}
    ]


for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

user_input = st.chat_input("你的问题/学习需求")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.spinner("AI正在思考中，请稍等..."):
        response = generate_response(
            user_input, prompt_template, st.session_state["memory"]
        )
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
