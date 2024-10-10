# LangChain_with_RAG
빠르게 랭체인 기반 RAG을 사용할 수 있습니다. 지원가능한 문서: TEXT, PDF, URL(일반). 이뿐만 아닌, 임베딩한 벡터저장소를 물리적으로 저장하고, 로드하여 언제든지 벡터저장소를 사용할 수 있습니다.

---

<br>

# 어떻게 사용하는가? ( 흐름설명 )

1.RAG 관리 인스턴스를 만들어봅시다!!
```python
inst = RAG_LLM()
```

2.최초 RAG을 위한 RAG체인을 만들어야합니다. 
```python
inst.Init_Rag_LLM('ABC', Ollama(base_url='http://192.168.0.100:11434', model='gemma2'), load_vectorstore=None)
# 'ABC'는 식별자 (식별자마다 별도의 체인을 생성합니다. )
#Ollama(base_url='http://192.168.0.100:11434', model='gemma2') 는 테스트한 LLM서버
# load_vectorstore=None 이 None이면 이전에 HDD에서 로드한 Chroma를 사용하지 않는다는 의미.
```

3.원하는 문서 (txt,pdf,url(일반request기반))를 읽고 단어를 분리하고 임베딩하여 벡터저장소에 "추가"하여 문서 컨텐츠를 "축적"합니다. 
```python
inst.TEXT_Rag('ABC', 'TEXT_RAG.txt')
inst.URL_Rag('ABC', 'https://ko.wikipedia.org/wiki/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD')
inst.PDF_Rag('ABC', './Ahnlab_EDR.pdf')
```

4.HDD에 임베딩된 "벡터저장소" 저장하기
```python
inst.Save_Vectorstore('ABC', 'ABC') # 저장 ( Chroma의 <from_texts> 메서드 기반으로 각 키를 분리하고, 하나로 저장 )
# 1. 인자: '식별자'
# 2. 인자: '세이브 id'
```

5.(4.)에 저장된 벡터저장소 로드하기
```python
loaded = inst.Load_Vectorstore('ABC')# 로드 ( Chroma 생성자에 persist_directory + 이전에 설정한 임베딩을 추가하여 로드함 )
# 1. 인자: '세이브 id'
```

6.새로운 식별자에 저장해서 바로 질의하기.
```python

inst.Init_Rag_LLM('DEF', Ollama(base_url='http://192.168.0.100:11434', model='gemma2'), load_vectorstore=loaded)  # load_vectorstore 에 로드한 Chroma넣기

r = inst.ASK_to_RAG_CHAIN('DEF', '한국에 대해서 한국어로 요약 설명해줘') # 바로 질의
print(r)
```
