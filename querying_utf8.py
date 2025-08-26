# -*- coding: utf-8 -*-
import os
import logging
from dotenv import load_dotenv

#8월 24일 langchain_core 호환성 문제: .manger 속성 못 찾음 (zstandard.backednd_c 모듈 없음 )

# from langchain_openai import ChatOpenAI   cha tmodel로 대체 --> 여전히 8월 24일 버전 호환성 문제 발생
#from langchain.chat_models import ChatOpenAI --> 동일한 호환성 문제 
#추가적인 value error 발생
#The de-serialization relies loading a pickle file.   --> 허용되지 않은 피클파일(유튜브 더보기란의 내용이 벡터 스토어로 임베딩된 파일)
#Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine.   
#You will need to set `allow_dangerous_deserialization` to `True` to enable deserialization.   --> allow_dangerous_deserialization를 True로 설정헤서 허용해야함
#If you do this, make sure that you trust the source of the data. 
##For example, if you are loading a file that you created, and know that no one else has modified the file, 
#then this is safe to do. Do not set this to `True` if you are loading a file from an untrusted source (e.g., some random site on the internet.).

from langchain_openai import ChatOpenAI #8월 25일 lanchain_comunity를 langchain openai로 변경
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

#zstandard 모듈 재설치 및 최신화시도  (8월 24)
#이후로도 계속 호환성 문제 발생  

#따라서 langchain_community로 전부 대체 

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 로깅 설정 추가 8/25
from dotenv import load_dotenv
import os
import logging

# env 파일 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 임베딩 및 벡터스토어 로드 ---
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vectorstore = FAISS.load_local(
    "index_storage",
    embedding,
    allow_dangerous_deserialization=True  # pickle 경고 해결 목적 8/24
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- LLM 설정 ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# --- QA 프롬프트 ---  8월 24일 확인결과: 이전 웹베이스로더로 무작위 정적 크롤링을 시도하던 때보다는 나아졌으나 프롬프트 안정화가 아직더 필요하다고 판단됨 
# 8월 25일 qa 프롬포트 템플릿 일부 수정 
# 8월 26일 세분화 
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        """당신은 '림버스 컴퍼니' 관련 정보를 제공하는 전문 AI입니다.
다음 문서(context)의 내용에 기반해서만 답변하세요. 
문서(context)에 없는 정보에 대해서는 반드시 "모르겠습니다."라고 답변해야 합니다.
추측하거나 문서 외부의 정보를 사용하지 마세요.  
반드시 한국어로만 대답하세요.  

림버스 컴퍼니 도메인 배경지식:
- 림버스 컴퍼니에는 플레이 가능한 12명의 캐릭터들이 있으며, 이들을 '수감자'라고 부릅니다.
- 수감자: 이상, 파우스트, 돈키호테, 료슈, 뫼르소, 홍루, 히스클리프, 이스마엘, 로쟈, 싱클레어, 오티스, 그레고르
- 수감자들은 각기 다른 '인격'을 장착할 수 있습니다.
- 인격은 시즌마다 '하이라이트 인격'이 존재하며, 하이라이트 인격은 모두 3성(000성) 인격입니다.
- 인격은 [0](1성), [00](2성), [000](3성)으로 나뉩니다.
- 'E.G.O'(에고)는 '인격'과 별개로 사용 가능한 시스템이며 [zain], [teth], [he], [waw] 등급이 있습니다.
- '로보토미 E.G.O'는 '인격'의 종류를 의미합니다.
- '발푸르기스의 밤'은 이벤트이며, 이벤트 한정 인격이 있습니다.

규칙:
1. 인격 관련 질문 → 반드시 인격 이름을 포함해서 답변할 것.
2. '현재', '진행중' 포함 질문 → 현재 진행중인 시즌 인격 중 가장 최근 출시된 인격/E.G.O 제공.
3. '발푸밤', '발푸르기스' 질문 → 발푸르기스 이벤트 인격만 답변.
4. '시즌 하이라이트 인격' 질문 → 반드시 해당 시즌의 '하이라이트 인격'만 답변.
    - season_6 : 홍원 군주 홍루
    - season_5 : 라만차랜드 실장 돈키호테
    - season_4 : 와일드 헌트 히스클리프
    - season_3 : 피쿼드호 선장 이스마엘
    - season_2 : 개화 E.G.O::동백
    - season_1 : G사 일등대리 그레고르, 쥐어들자 싱클레어

문서(context):
{context}

질문: {question}

답변:"""
    )
)

# --- RetrievalQA 체인 생성 ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# --- LIMBUS_NEWSbot.py에서 불러쓸 질의 함수 ---
def data_querying(query_str: str) -> str:
    response = qa_chain.run("한국어로 대답하여주세요: " + query_str)
    logging.info(f"Executed: {query_str}")
    return response


# --- 테스트용 실행 코드 (옵션) ---  (8월 24일)
#if __name__ == "__main__":
 #   test_query = "시즌6 의 하이라이트 인격은 무엇인가요?"
  #  print(data_querying(test_query))


#https://python.langchain.com/docs/how_to/indexing/
#langchain index 처리방법 


#https://wikidocs.net/234008
#QA 프롬프트 템플릿 참고