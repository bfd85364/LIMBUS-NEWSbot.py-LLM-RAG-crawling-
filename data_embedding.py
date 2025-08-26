# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import logging
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_yt_dlp import YoutubeLoaderDL
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # LangChain용 임베딩 --> 더는 llama index 호환  안할거임  

#임시 파싱  url (현재는document_loader중 YoutubeLoaderDL을 활용한  정적크롤링) 
#차후 추가할 코드로는 일단, py.pdf 내용을 메인으로 활용하는 방안으로 대체 

#정적크롤링을 통하여 영상 설명 더보기란의 인격정보 (인격명, 몇성인지 ex) 0성, 00성 000성, 시즌별 인격, 시즌 하이라이트 인격 구분) 가져옴
#영상이 업데이트 마다 season 변수의 리스트를 추가하는 형식으로 업데이트할 예정 
#문제는 발푸밤 인격정보는 발푸밤 이벤트 pv 영사에 껴서 나오는데, 발푸밤 pv의 영상 설명더보기란의 경우 인격정보가 없음(공지 X)               
     
#인게임의 클라이언트 뜯기는 시도하지 않을 예정 -> 스토리 스포일러 정보와 회사 내 민감정보가 많음, 팬덤 내에서 클라이언트를 뜯는 행위를 달가워하지 않음

#season 6 URL
season_6 =["https://www.youtube.com/watch?v=MdZthRqLL6Q", #시즌 6 하이라이트 인격
           "https://www.youtube.com/watch?v=-pOx9fo6A7M",
           "https://www.youtube.com/watch?v=wbu2viInM3w",
           "https://www.youtube.com/watch?v=988V0rCkwc4",
           "https://www.youtube.com/watch?v=Q8aayN94mAE",
            "https://www.youtube.com/watch?v=JYqUwRCEXMw"]


# 시즌 5 URL
season_5 = [
    "https://www.youtube.com/watch?v=YaZ2FlhqvI0",
    "https://www.youtube.com/watch?v=vBob-jTmlnc",
    "https://www.youtube.com/watch?v=pEDkIzAgXRk",  # E.G.O 정보 포함
    "https://www.youtube.com/watch?v=hdoQIlhjd7c",  # 하이라이트 인격
    "https://www.youtube.com/watch?v=oOXdURf30HE",  # E.G.O 정보 포함
    "https://www.youtube.com/watch?v=6xvYcgUMyKo",  # E.G.O 정보 포함
    "https://www.youtube.com/watch?v=p1QAz4OdjR4",
    "https://www.youtube.com/watch?v=ct5DyDdGjhs"
]

# 시즌 4 URL
season_4 = [
    "https://www.youtube.com/watch?v=Svaaq9F7Ty8",
    "https://www.youtube.com/watch?v=iv9qlLoU_dU",  # E.G.O 정보 포함
    "https://www.youtube.com/watch?v=h4_Vr90IKws",  # 하이라이트 인격
    "https://www.youtube.com/watch?v=Ibkwc5L7-ao",  # E.G.O 정보 포함
    "https://www.youtube.com/watch?v=CDD9KOauxM0",  # E.G.O 정보 포함
    "https://www.youtube.com/watch?v=UwQpka7htr4",
    "https://www.youtube.com/watch?v=4wpmk1qVK5k",
    "https://www.youtube.com/watch?v=v80gG3XgHPQ",
    "https://www.youtube.com/watch?v=H7jd3lnGIGA"
]

 #시즌 3 URL
season_3 = ["https://www.youtube.com/watch?v=W8MHensC0VU",
            "https://www.youtube.com/watch?v=I2ytdOuizWo",
            "https://www.youtube.com/watch?v=wRBvmbCcciI", #시즌 3 하이라이트 인격
            "https://www.youtube.com/watch?v=KK8Ziib6d48",
            "https://www.youtube.com/watch?v=hMeLXsKBPVY",
            "https://www.youtube.com/watch?v=rmePzJHACto",
            "https://www.youtube.com/watch?v=rZJhsuvvltk",
            "https://www.youtube.com/watch?v=0Y_mqN117Qs",
            "https://www.youtube.com/watch?v=5IqXi_SPSn0"
]

#시즌 2 URL
season_2 =["https://www.youtube.com/watch?v=hFnOO3abFHk",
          "https://www.youtube.com/watch?v=ntQTW6DF30g",
          "https://www.youtube.com/watch?v=8GkXAjGKBF4",
          "https://www.youtube.com/watch?v=0I_Rygsi17g"
]



# ==========================
# 환경 변수 로드 및 로깅 설정
# ==========================
load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# ==========================
# 시즌별 영상 로딩 함수
# ==========================
def load_season_docs(season_urls, season_name):
    print(f"=== {season_name} 로드 시작 ===")
    docs = []

    for idx, url in enumerate(season_urls, start=1):
        try:
            loader = YoutubeLoaderDL.from_youtube_url(url, add_video_info=True)
            video_docs = loader.load()

            for d in video_docs:
                docs.append(d)

                if "description" in d.metadata and d.metadata["description"]:
                    desc_doc = Document(
                        page_content=d.metadata["description"],
                        metadata={
                            "source": url,
                            "season": season_name,
                            "type": "video_description",
                            **d.metadata,
                        },
                    )
                    docs.append(desc_doc)

            print(f"[{season_name}] {idx}/{len(season_urls)} 완료: {url}")
        except Exception as e:
            print(f"[{season_name}] {idx}/{len(season_urls)} 실패: {url} ({e})")

        time.sleep(random.uniform(3, 5))

    print(f"=== {season_name} 로드 완료: {len(docs)}개 문서 ===")
    return docs


# ==========================
# 전체 시즌 문서 로드
# ==========================
all_docs = []
all_docs.extend(load_season_docs(season_6, "Season 6"))
all_docs.extend(load_season_docs(season_5, "Season 5"))
all_docs.extend(load_season_docs(season_4, "Season 4"))
all_docs.extend(load_season_docs(season_3, "Season 3"))
all_docs.extend(load_season_docs(season_2, "Season 2"))


# ==========================
# 문서 분할 (청크)
# ==========================
if not all_docs:
    print("\n오류: 문서를 로드하지 못했습니다. YouTube URL이나 로더 설정을 확인하세요.")
    texts = []
else:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(all_docs)
    print(f"\n총 {len(texts)}개의 텍스트 청크로 분할되었습니다.")


# ==========================
# FAISS 벡터스토어 생성 및 저장
# ==========================
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask") #사용자에게 한국어로 잡변을 제공해야하므로, 또한 사용하는 문서는 영문명과 한글명이 혼재되어있음 -> 이중에서 한국어만 추출되도 상관x 

vectorstore = FAISS.from_documents(texts, embedding)
vectorstore.save_local("index_storage") #index_storage 폴더에 저장  (로컬 저장)  

logging.info("모든 시즌 유튜브 문서가 임베딩 및 저장되었습니다.")


#https://velog.io/@sewer0pipe/Python-%EB%94%94%EC%8A%A4%EC%BD%94%EB%93%9C-%EC%9C%A0%ED%8A%9C%EB%B8%8C-%EC%9D%8C%EC%95%85%EC%9E%AC%EC%83%9D-%EB%B4%87-%EB%A7%8C%EB%93%A4%EA%B8%B0-discord-music-bot 
#디코봇 youtubeDL 활용 처리 관련 참고

#https://wikidocs.net/234008
#청크 분할 관련 참고

#url 로드시 3분 이상 소요 (대략 4분)