from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from memory import memoryStore
from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime
import requests
import os
from dotenv import load_dotenv


class OpenAIAgent:

    def __init__(self) -> None:
        load_dotenv()

        

    async def get_ai_response(self, user_input: str, unique_session_id: str): 

        if(unique_session_id not in memoryStore):
            memoryStore[unique_session_id] = {
                "memory": ConversationBufferWindowMemory(output_key="output", memory_key="chat_history", k=10, return_messages=True),
                "last_accessed": datetime.now()  
            }
        else:
            memoryStore[unique_session_id]["last_accessed"] = datetime.now()

        Unique_memory = memoryStore[unique_session_id]["memory"] 

        openai_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=os.getenv("Open_api_key"))

        system = '''Respond to the human as helpfully and accurately as possible. You have access to the following tools:

                    {tools}

                    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

                    Valid "action" values: "Final Answer", "fetch_weather", {tool_names}

                    Provide only ONE action per $JSON_BLOB, as shown:

                    ```
                    {{
                    "action": $TOOL_NAME,
                    "action_input": $INPUT
                    }}
                    ```

                    Follow this format:

                    Question: input question to answer
                    Thought: consider previous and subsequent steps
                    Action:
                    ```
                    $JSON_BLOB
                    ```
                    Observation: action result
                    ... (repeat Thought/Action/Observation N times)
                    Thought: I know what to respond
                    Action:
                    ```
                    {{
                    "action": "Final Answer",
                    "action_input": "Final response to human"
                    }}

                    Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB``` then Observation.
                    
                    '''

        human = '''

        Previous_conversation: {chat_history}

        Question: {input}

        Thought: {agent_scratchpad}

        (reminder to respond in a JSON blob no matter what)

        '''

        full_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                template=system,
                input_variables=['tool_names', 'tools']
            ),
            MessagesPlaceholder(
                variable_name= "chat_history", optional=True
            ),
            HumanMessagePromptTemplate.from_template(
                template=human,
                input_variables=['input', 'agent_scratchpad', 'chat_history']
            ),
        ])

        @tool
        def fetch_weather(city: str) -> str:
            """Fetches weather data for a specified city. Temperature will be in Kelvin"""
            api_key = "4ec23faf1ddad6b88d896324f5afd2de"
            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            complete_url = f"{base_url}appid={api_key}&q={city}"
            response = requests.get(complete_url)
            weather_data = response.json()

            if weather_data["cod"] != "404":
                return str(weather_data)
            else:
                return "City not found. Please check the city name."
            

        # print(fetch_weather("Chennai"))

        @tool
        def addition_funct(a: int, b: int) -> str:
            '''Use this function to add two numbers'''
            return str(a + b)

        @tool
        def multiply(a: int, b: int) -> str:
            '''Use this function to multiply two numbers'''
            return str(a * b)

        tools_list = [addition_funct, multiply, fetch_weather]


        open_ai_agent = create_structured_chat_agent(openai_model, tools_list, full_prompt)

        open_ai_agent_executor = AgentExecutor(agent=open_ai_agent, tools=tools_list, max_iterations=10, verbose=True, handle_parsing_errors=True, memory=Unique_memory)

        buffer = ""
        action_input_start = False
        final_answer_found = False

        async for event in open_ai_agent_executor.astream_events({"input": user_input}, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                buffer += content  # Append the incoming stream content to the buffer

                # Check for the start of the "Final Answer" section
                if "Final Answer" in buffer and not final_answer_found:
                    final_answer_found = True
                    buffer = ""

                if final_answer_found:
                    # Check for the start of the "action_input" segment
                    if '"action_input":' in buffer and not action_input_start:
                        start_index = buffer.find('"action_input":') + len('"action_input":') + 1
                        action_input_start = True
                        buffer = buffer[start_index:]  # Remove everything before "action_input"

                    # If we're within the "action_input" section, start yielding
                    if action_input_start:
                        closing_index = buffer.find('}')
                        if closing_index != -1:
                            # Stop yielding if we've reached the end of "action_input"
                            action_input_content = buffer[:closing_index].strip('"')
                            yield action_input_content
                            buffer = ""
                            final_answer_found = False  # Reset for next potential "Final Answer"
                            action_input_start = False
                        else:
                            # Yield the current buffer content as it's part of "action_input"
                            action_input_content = buffer.strip('"')
                            print(action_input_content, end="", flush=True)
                            yield action_input_content
                            buffer = ""  # Clear buffer to avoid re-yielding the same content