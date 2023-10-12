from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import DeepSparse

from pytube import YouTube
import whisper
import tempfile

MODEL_PATH = "zoo:mpt-7b-gsm8k_mpt_pretrain-pruned60_quantized"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
llm = DeepSparse(model=MODEL_PATH)
model = whisper.load_model("base")

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
Example of your response should be:
```
The answer is foo
SOURCES: xyz
```
Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

def transcribe(youtube_url, model):
    youtube = YouTube(youtube_url)

    audio = youtube.streams.filter(only_audio=True).first()

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        title = os.path.basename(file)[:-4]
        result = model.transcribe(file, fp16=False)

    return title, youtube_url, result["text"].strip()
    
@cl.on_chat_start
async def init():
    url = None

    # Wait for the user to upload a file
    while url == None:
        url = await cl.AskUserMessage(content="Please type a YouTube URL to begin!").send()

    msg = cl.Message(content=f"Processing video...")
    await msg.send()
        
    transcriptions = transcribe(str(url), model)
        
    texts = text_splitter.create_documents([transcriptions[2]])
    for i, text in enumerate(texts): text.metadata["source"] = f"{i}-pl"
        
    # Create a Chroma vector store
    docsearch = Chroma.from_documents(texts, embeddings)
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        return_source_documents=True,
        retriever=docsearch.as_retriever(),
    )

    # Save the metadata and texts in the user session
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    msg.content = f"Processing `{transcriptions[0]}` video done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["result"]
    source_documents = res["source_documents"]
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if source_documents:
        found_sources = []

        # Add the sources to the message
        for source_idx, source in enumerate(source_documents):
            # Get the index of the source
            source_name = f"source_{source_idx}"
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=str(source.page_content).strip(), name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"
            
    if cb.has_streamed_final_answer:
        cb.final_stream.content = answer
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer,
                         elements=source_elements
                        ).send()
