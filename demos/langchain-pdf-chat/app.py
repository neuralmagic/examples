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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
import PyPDF2

MODEL_PATH = "hf:neuralmagic/mpt-7b-chat-pruned50-quant"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
llm = DeepSparse(model=MODEL_PATH)

@cl.on_chat_start
async def init():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!", accept=["application/pdf"], max_size_mb=50,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")
        
    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
        
    # texts = text_splitter.create_documents(pdf_text)
    texts = text_splitter.create_documents([pdf_text])
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
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
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
