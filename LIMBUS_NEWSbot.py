# -*- coding: utf-8 -*-

# 8월 26일 
#https://interactions-py.github.io/interactions.py/Guides/03%20Creating%20Commands/ 
#discord-py-interctions  slash command 사용법 

#https://github.com/aio-libs/aiohttp/issues/4158
#aiohttp 에러 

#(env-LCB) PS C:\Users\User\Desktop\전공\python\LIMBUS-NEWSbot> pip install discord-py-interactions==5.15.0
#ERROR: Ignored the following versions that require a different python version: 5.0.0 Requires-Python >=3.10; 5.0.1 Requires-Python >=3.10; 5.1.0 Requires-Python >=3.10; 5.10.0 Requires-Python >=3.10; 5.11.0 Requires-Python >=3.10; 5.12.0 Requires-Python >=3.10; 5.12.1 Requires-Python >=3.10; 5.13.0 Requires-Python >=3.10; 5.13.1 Requires-Python >=3.10; 5.13.2 Requires-Python >=3.10; 5.14.0 Requires-Python >=3.10; 5.15.0 Requires-Python >=3.10; 5.15.0rc1 Requires-Python >=3.10; 5.2.0 Requires-Python >=3.10; 5.3.0 Requires-Python >=3.10; 5.3.1 Requires-Python >=3.10; 5.4.0 Requires-Python >=3.10; 5.5.0 Requires-Python >=3.10; 5.5.1 Requires-Python >=3.10; 5.6.0 Requires-Python >=3.10; 5.7.0 Requires-Python >=3.10; 5.8.0 Requires-Python >=3.10; 5.9.0 Requires-Python >=3.10; 5.9.1 Requires-Python >=3.10; 5.9.2 Requires-Python >=3.10
#ERROR: Could not find a version that satisfies the requirement discord-py-interactions==5.15.0 (from versions: 3.0.2, 4.0.0, 4.0.1, 4.0.2, 4.1.0, 4.2.0, 4.2.1, 4.3.0, 4.3.1, 4.3.2, 4.3.3, 4.3.4, 4.4.0, 4.4.1)
#ERROR: No matching distribution found for discord-py-interactions==5.15.0

#discord-py-interactions의 v5는 현재 파이썬 10버전 이상 부터 사용가능
#따라서 기종 3.9환경의  env-LCB 가상환경 삭제후 새로운 셋팅 필요 
 
import interactions
from interactions import Client, slash_command, SlashContext, slash_option, OptionType #from interactions import listen 사용 지양
from dotenv import load_dotenv
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from querying_utf8 import data_querying


# Load .env file
load_dotenv()

# Get the bot token from environment variables
TOKEN = os.getenv("DISCORD_BOT_TOKEN")


# Set up logging
logging.basicConfig(level=logging.INFO)

# Create the bot client instance
# test_guilds now accepts a list of IDs
bot = Client(token=TOKEN, intents=interactions.Intents.ALL, default_scope="139279505651099660")
executor = ThreadPoolExecutor(max_workers=3)   #8월 36일 Intents 직접 사용대신 intrections.Intents.ALL로 표현 

@bot.listen()
async def on_ready():
    print("Bot is ready.")

# --- Slash Command: /query  수정 8월 26일---
@slash_command(name="query", description="질문을 입력해주세요")
@slash_option(
    name="input_text",
    description="입력할 질문",
    required=True,
    opt_type=OptionType.STRING,
)
#8월 26일
async def get_response(ctx: SlashContext, input_text: str):
    await ctx.defer()
    try:
        loop = asyncio.get_event_loop()
        # data_querying 을 별도 쓰레드에서 실행
        result = await loop.run_in_executor(executor, data_querying, input_text)
       # formatted = f" 답변: {result}" ---> ctx 방식으로바뀜 
        await ctx.send(f"답변: {result}")

    except Exception as e:
        logging.error("data_querying.py에서 실행중 오류:", exc_info=True)
        formatted = f"오류가 발생하였습니다.: {e}"
        await ctx.send("오류 발생")


# 프로그램 진입점에서만 실행
if __name__ == "__main__": 
    bot.start(TOKEN)



    #파이썬 버전 업데이트 이후 필요한 모듈들 
     #Discord 봇 (v5)
     #discord-py-interactions==5.15.0
    #aiohttp>=3.9.5
    #attrs>=25.3.0

# 환경 변수 로딩
#python-dotenv>=1.1.1

# LangChain / HuggingFace 임베딩
#langchain>=0.3.27
#langchain-community>=0.3.27
#langchain-huggingface>=0.3.1
#langchain-openai>=0.3.31
#langchain-core  
#sentence-transformers
#faiss -cpu

# 벡터 DB
#faiss-cpu>=1.12.0

# 유튜브 url정적 크롤링 / 파싱
#beautifulsoup4>=4.13.4 -차후 단빵숲 활용시 추가 
#lxml>=6.0.0- 차후 당 빵숲 활용시추가  
#requests>=2.32.4 #차후, 단빵숲 활용시 추가 
#yt-dlp==2025.8.20
#pytube>=12.1.2



#에러: <aiohttp.client.ClientSession object at 0x...>" 발생 
#Discord 봇이 실행을 마친 후에도 aiohttp.ClientSession 객체가 제대로 종료되지 않아 발생하는 
#'미해결 클라이언트 세션(Unclosed client session)' 오류 메시지입니다. 
#이는 봇 프로그램이 종료되기 전에 세션이 깔끔하게 닫히지 않았을 때 나타나며, 일반적으로 `client.run()`을 호출한 후 봇이 예기치 않게 종료되거나, discord.Client 객체를 제대로 정리하지 못해 발생


#8월 25일 
#예외처리 추가 : 실수로 env 파일을 훼손했을것을 대비
#if TOKEN is None:
#raise ValueError("DISCORD_BOT_TOKEN이 .env에서 로드되지 않았습니다. .env 파일을 확인하세요.")

# 이전 코드에서 @bot.command() 방식 사용
#그러나 실제로 최신 버전 권장 방식은 slash_command 사용임 

# 슬래시 커맨드 정의
#@interactions.slash_command(
   # name="ask",
 #   description="림버스 관련 질문하기",
#)
#@interactions.option(
 #   name="question",
   # description="질문을 입력하세요",
  #  type=interactions.OptionType.STRING,
    #required=True,
#)
#async def ask(ctx: interactions.SlashContext, question: str):
    # querying_utf8.py의 data_querying 함수를 호출하여 답변을 생성
    #await ctx.defer() # 답변 생성에 시간이 걸릴 수 있으므로 '봇이 생각 중...' 상태를 표시
   # try:
   #     answer = data_querying(question)
   #     await ctx.send(answer)
  #  except Exception as e:
        # 오류 발생 시 사용자에게 메시지를 보냄
 #       await ctx.send(f"오류가 발생했습니다: {e}")

# 봇 실행
#if __name__ == "__main__":
    # interactions 라이브러리에서 권장하는 실행 방식으로 수정
    # asyncio.run(bot.start()) 대신 bot.start()를 직접 호출
   # bot.start()

    #이전 방식 -> 봇 실행마다 인덱스를 계속 재생성하는 구조--> 비용적인 측면에서 비효율적 
    #비용 소모를 줄이고자 openai의 임베딩 모델을 langchain의 LangchainEmbedding으로 변경(더이상 llama_index와 호환 x))
    #따라서 현재는 한번 생성한 인덱스를 로드하는 방식으로 변경 --> 이렇게 하면 매 실행시마다 비용과 실행시간을 절약 할수 있음. 
    #이후 py.pdf 문서를 메인으로 활용하는 방안으로 대체후 추가로 내용이 없데이트 될때마다 index_storage를 업데이트하는 방안으로 변경하는 방법 구축할 예정 


    #동기식: 봇이 응답을 기다리는 동안 다른 작업을 수행하지 못함 (여러 작업을 하나씩 절차적으로 처리)
    #비동기식: 봇이 응답을 기다리는 동안에도 다른 작업을 수행할 수 있음 (여러 작업을 동시에 처리)
    #asyncio는 비동기 프로그래밍을 위한 모듈

    #asyncio는 이벤트 루프와 코루틴을 기반으로 동작하며 데이터를 요청하고 응답을 기다리는 I/O bound한 작업에서 효율적입니다.
    #코루틴 기반이므로 멀티 스레드와 비교하여 문맥교환에 따른 비용이 다소 적게 들어갑니다.

    #await 뒤에 time.sleep과 같이 사용한다면 스레드가 중단되므로 의도한대로 동작하지 않습니다. 
    ##따라서 코루틴으로 구현되어 있는 asyncio.sleep을 사용해야 합니다. 
    #코루틴으로 만들어진 모듈이 아니라면 (requests, psycopg, django orm 등등) await을 붙여도 소용 X
    #여기서 코루틴이란 용어는 코루틴 함수와 코루틴 객체라는 두 의미를 내포하는 것을 알 수 있습니다.
    #코루틴 함수: async def ~~ 로 정의된 함
    #코루틴 객체: 코루틴 함수를 호출하고 반환되는 객체
    #asyncio는 이전 버전인 제너레이터 기반 코루틴도 지원한다고 하지만 3.10부터는 사라질 기술이므로 사용하지 않는 것이 좋아 보입니다.

    #참고자료: 
    # https://docs.python.org/ko/3.8/library/asyncio.html
    # https://brownbears.tistory.com/54e

    #8월 24일 
#import interactions --> 이전처럼 또다시 interactions 모듈을 인지하지 못하는 문제 발생
#from discord_interactions import Intents, slash_command, SlashContext   #discord_interactions로 모듈명 변경
#from discord import Client, Intents
#discord.py 라이브러리에서 Client 임포트 ---> interactions 모듈 대신 사용 discord.py로 전부 통일
#문제는 slash_command, slashContext가 discord.py에 없음 --> discord-py-slash-command 설치 하기   --> 문제 발생 
#discord 모듈과 dicord interactions도 같이 입력하여 slash_command, SlashContext를 사용하면서, Client 클래스도 사용하려고 하였으나 discord.py와 충돌 발생

#8월 25일 
#이전 interactions 모듈로 다시 변경
#--> 이후에도 runtime error 발생함.  (#no qa))


#https://stackoverflow.com/questions/45346575/what-does-noqa-mean-in-python-comments

#Adding  noqa to a line indicates that the linter 
#(a program that automatically checks code quality) should not check this line. Any warnings that code may have generated will be ignored.
#That line may have something that "looks bad" to the linter, but the developer understands and intends it to be there for some reason

# noqa → Flake8 같은 린터가 이 줄은 검사하지 말라는 지시

#F401 → "임포트했는데 안 쓰고 있다" 경고 무시

#F403 → "import *은 권장되지 않는다" 경고 무시

#https://gencomi.tistory.com/entry/asyncio%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%9C-%EB%B9%84%EB%8F%99%EA%B8%B0-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-asyncioensurefuture%EC%99%80-loopcreatetask%EC%9D%98-%EC%B0%A8%EC%9D%B4%EC%A0%90
#runtime error의 원인이 코루틴과 관련됨을 확인 
#정확히는 현재 비동기식의 코드에 동기식 모듈을 사용한것이 오류의 원인
#--> (interactions(=discord-py-interactions) 는 내부적으로 aiohttp.ClientSession()을 비동기 루프 안에서 실행해야 하는 객체로 만들어야 하는데) 해당 모듈이 aiohttp.ClientSession()을 import 시점에서 바로 실행
#(동시 실행 불가)

#따라서 코루틴  형식으로 변형 

#그럼에도 지속적으로 runtime error 발생
#우선, pip 모듈중 discord-py-interactions 모듈을 업그레이드 시도해봄 

#Traceback (most recent call last):
 # File "C:\Users\User\Desktop\전공\python\LIMBUS-NEWSbot\LIMBUS_NEWSbot.py", line 46, in <module>
 #   from interactions import (
#ImportError: cannot import name 'slash_command' from 'interactions' (C:\Users\User\Desktop\전공\python\LIMBUS-NEWSbot\env-LCB\lib\site-packages\interactions\__init__.py)
#Unclosed client session
#client_session: <aiohttp.client.ClientSession object at 0x0000021E0F2E39A0>

#이후 출력 결과를 보니 interactions 모듈에서 slash_command를 임포트 하지 못하는 문제 발생
