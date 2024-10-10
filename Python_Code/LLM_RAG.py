import time
from typing import Any, Union, Optional

from langchain.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms.ollama import Ollama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAG_LLM():
    def __init__(self):
        self.Rag_LLM = {}

    # LLM 최초 등록
    def Init_Rag_LLM(self, RAG_ID:str, LLM_Model:Any, load_vectorstore:Chroma = None):
        if(self.Check_exists_Rag_LLM(RAG_ID)):
            return False

        embeddings = GPT4AllEmbeddings()
        if(load_vectorstore): # 이미 저장소가 있는 경우.
            vectorstore = load_vectorstore
        else:
            vectorstore = Chroma(embedding_function=embeddings)

        RAG_CHAIN =  RetrievalQA.from_chain_type(
            llm=LLM_Model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        self.Rag_LLM[RAG_ID] = {
            'RAG_CHAIN':RAG_CHAIN,
            'vectorstore':vectorstore,
            'emmadding':embeddings
        }
        #print(self.Rag_LLM[RAG_ID])

    # URL
    def URL_Rag(self, RAG_ID:str, url:str, chunk_size:int=500, chunk_overlap:int=0)->bool:
        if(not self.Check_exists_Rag_LLM(RAG_ID)):
            return False

        return self.Add_info(RAG_ID, WebBaseLoader(web_path=url), chunk_size, chunk_overlap)

    # TEXT
    def TEXT_Rag(self, RAG_ID:str, text_path:str, chunk_size:int=500, chunk_overlap:int=0)->bool:
        if (not self.Check_exists_Rag_LLM(RAG_ID)):
            return False

        return self.Add_info(RAG_ID, TextLoader(file_path=text_path,encoding='UTF-8'), chunk_size, chunk_overlap)

    # PDF
    def PDF_Rag(self, RAG_ID:str, pdf_path:str, chunk_size:int=500, chunk_overlap:int=0)->bool:
        if (not self.Check_exists_Rag_LLM(RAG_ID)):
            return False

        return self.Add_info(RAG_ID, PyPDFLoader(file_path=pdf_path), chunk_size, chunk_overlap)

    def Add_info(self, RAG_ID:str, loader:Union[WebBaseLoader, TextLoader, PyPDFLoader], chunk_size:int=500, chunk_overlap:int=0)->bool:
        if(not self.Check_exists_Rag_LLM(RAG_ID)):
            return False

        info_content = loader.load()

        # 텍스트를 청크로 나눔
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        split_texts = text_splitter.split_documents(info_content)
        #print(f"텍스트 나누기 완료, 나누어진 텍스트 갯수: {len(split_texts)} {split_texts}")

        # 벡터 저장소에 문서 추가
        self.Rag_LLM[RAG_ID]['vectorstore'].add_documents(split_texts)

        return True

    def ASK_to_RAG_CHAIN(self, RAG_ID:str, Input:str):
        if(not self.Check_exists_Rag_LLM(RAG_ID)):
            return None

        return self.Rag_LLM[RAG_ID]['RAG_CHAIN'].invoke({"query": Input})






    def Save_Vectorstore(self, RAG_ID:str, SAVE_id:str)->bool:
        if(not self.Check_exists_Rag_LLM(RAG_ID)):
            return False

        import pickle, zlib

        data = self.Rag_LLM[RAG_ID]['vectorstore']
        #print(data)

        all_data = data._collection.get()

        existing_texts = all_data['documents']
        # metadatas 가져오기 (있는 경우)
        existing_metadatas = all_data['metadatas']

        embeddings = GPT4AllEmbeddings()
        new_db = Chroma.from_texts(
            texts=existing_texts,
            embedding=embeddings,
            persist_directory=f"./{SAVE_id}__chroma_db",
            metadatas=existing_metadatas,
        )
        return True

    def Load_Vectorstore(self, SAVE_id:str)->Optional[Chroma]:
        loaded_db = None
        embeddings = GPT4AllEmbeddings()
        loaded_db = Chroma(
            persist_directory=f"./{SAVE_id}__chroma_db",
            embedding_function=embeddings
        )
        ##print(loaded_db)
        # 3. 로드된 데이터 확인
        data = loaded_db._collection.get()
        #print(data)

        return loaded_db


    def Check_exists_Rag_LLM(self, RAG_ID:str)->bool:
        if(RAG_ID in self.Rag_LLM):
            return True
        else:
            return False

inst = RAG_LLM()

inst.Init_Rag_LLM('ABC', Ollama(base_url='http://192.168.0.100:11434', model='gemma2'), load_vectorstore=None)

inst.TEXT_Rag('ABC', 'TEXT_RAG.txt')
inst.URL_Rag('ABC', 'https://ko.wikipedia.org/wiki/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD')
#r = inst.ASK_to_RAG_CHAIN('ABC', '한국에 대해서 한국어로 요약 설명해줘')
#print(r)

inst.Save_Vectorstore('ABC', 'ABC') # 저장

#------

loaded = inst.Load_Vectorstore('ABC')# 로드
inst.Init_Rag_LLM('DEF', Ollama(base_url='http://192.168.0.100:11434', model='gemma2'), load_vectorstore=loaded)

r = inst.ASK_to_RAG_CHAIN('DEF', '한국에 대해서 한국어로 요약 설명해줘')
print(r)


