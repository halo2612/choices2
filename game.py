from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain_community.chat_message_histories import CassandraChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
# from datasets import load_dataset

import json

with open("choices2-token.json") as f:
    secrets = json.load(f)

cloud_config = {
    'secure_connect_bundle': 'secure-connect-choices2.zip'
}

auth_provider = PlainTextAuthProvider(
    secrets["clientId"],
    secrets["secret"]
)

cluster = Cluster(
    cloud=cloud_config,
    auth_provider=auth_provider
)

session = cluster.connect()

# ds = load_dataset("rohitsaxena/MovieSum", split="train", streaming=True)

# print(next(iter(ds)))

message_history = CassandraChatMessageHistory(
    session_id="anything",
    session=session,
    keyspace="default_keyspace",
    ttl_seconds=3600
)

message_history.clear()

cass_buff_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history
)

# def get_movie_summary(movie_name):
#     # Find the movie summary in the dataset
#     movie_summary = next((item for item in ds if item["movie_name"] == movie_name), None)
#     return movie_summary['summary']

# movie_name = input("Enter a movie name: ")
# movie_summary = get_movie_summary(movie_name)

# if movie_summary is None:
#     print("Movie not found!")
#     exit(1)

template = """
In the near future, humanity is facing extinction following a global famine caused by ecocide. Cooper's family, which includes his children, Tom and Murph, and his father-in-law Donald, engage in farming, like most of humanity. One evening, during a dust storm, Cooper and Murph discover a gravitational anomaly in Murph's bedroom that left patterns of falling particles. The pattern resolves into GPS coordinates, which lead Cooper and Murph to the secret facility of NASA, believed to have shut down. NASA's mission is to find a habitable planet beyond the solar system. A team is preparing to travel through a wormhole near Saturn leading to a galaxy with three potentially habitable planets, based on data from three probe missions, each orbiting the supermassive blackhole Gargantua. Dr. Brand, lead scientist, asks Cooper to pilot due to his experience. Cooper struggles with leaving his children behind, but promises Murph that he will return. Heartbroken, Murph refuses to say goodbye. The Endurance spacecraft takes off with three other NASA scientists − Romilly, Doyle, and Brand's daughter Amelia − as well as two intelligent robots, TARS and CASE. The crew arrive at the first planet to find an ocean world with massive tidal waves. Doyle is swept away and killed. Cooper and Amelia wait for water to drain from the craft and return to the Endurance, where 23 years have passed for Romilly due to time dilation. On Earth, an adult Murph helps Dr. Brand with an equation to solve the problem of mass exodus — transporting Earth's population off the planet. On his deathbed, Brand reveals that the crew's goal was to colonize another planet rather than return with data that would help the existing population. Murph, despite wondering if Cooper knowingly abandoned her, continues to work on the problem. On the second planet, the crew awakens the probe mission's explorer, Dr. Mann, from cryostasis. Mann offers to show Cooper the hospitable part of the planet, but then confesses to having falsified data so that someone might rescue him from the deserted ice world. Cooper survives the fight, while Romilly is killed while accessing Mann's data. Mann takes the Endurance crew's lander and beats them to the Endurance. With TARS having removed his security clearance to dock, Mann fails to dock correctly and is killed in an explosion that damages a portion of Endurance. Cooper and Amelia, with limited fuel remaining, chart a gravity-assist path around Gargantua to propel the craft to the third planet. At the last moment, Cooper detaches himself so that Amelia may reach the final planet. He and TARS fall beyond the black hole's event horizon. Cooper falls into a five-dimensional tesseract, with time as a physical dimension, that appears to be the back of Murph's bookshelf. He uses gravity to communicate with Murph and his past self, and realizes he sent the NASA coordinates and initiated this mission, and that a future generation created the tesseract to preserve humankind. He decides to feed Murph the blackhole gravity data to help her solve the equation, and encodes the data as Morse code into the ticking minute hand of his wristwatch, which he gave Murph. Murph comes across the watch while visiting the family home for the last time. The tesseract dissolves, ejecting Cooper and TARS through the wormhole and into Saturn's orbit again, where a present-day (22nd century CE) spacefaring O'Neill cylinder picks them up. Cooper awakens in a hospital in a space colony and is reunited with a dying Murph, now considerably older than him. Murph asks Cooper not to watch her die and to go to Amelia. Cooper and TARS set off for the third planet, where Amelia is directing the setup of a new colony. She takes off her helmet and breathes in the air.

Based on the summary of this movie your goal is to create a branching narrative experience where each choice
leads to a new path, ultimately determining User's fate.

Here are some rules to follow:
1. Start by asking the player to choose which character they want to play.
2. Have a few paths that lead to success
3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game
4. Make sure that the user makes atleast 20 choices before the game ends whether it be success or failure.

Here is the chat history, use this to understand what to say next: {chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

llm = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=cass_buff_memory
)

choice = "start"

while True:
    choice = input("Your reply: ")

    response = llm_chain.predict(human_input=choice)
    print("response: ", response)

    if "The End." in response:
        break