from openai import OpenAI
import os

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

def reply_by_chatgpt(post):
    api_key = ""
    api_base = "https://api.xi-ai.cn/v1"
    client = OpenAI(api_key=api_key, base_url=api_base)

    prompt = """你是一个富有同理心的好心人，下面是网友对自己情况的描述：
    {{post}}

    请你以一名普通人的口吻回复这位网友。你要用到以下策略：与网友产生情绪支持与共鸣：侧重于回复者对原帖作者情绪内容的理解、支持和共鸣，提供安慰或鼓励。回复者在此类回复中通常会表达对原帖作者情绪的理解和支持，以减轻其负面情绪，让其感到被理解和关心。这类回复更注重情绪支持和情感理解。
    你的回复字数在140字以内，不要生成emoji。
    """.strip()
    # try_num = 5
    # while try_num:
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.replace('{{post}}', post)}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        # try_num -= 1
    
    return ""
        
# 读取 contents.txt 中的内容
with open('data/contents.txt', 'r', encoding='utf-8') as file:
    posts = file.readlines()

# 生成回复并输出到 comments.txt 中
with open('data/comments.txt', 'w', encoding='utf-8') as output_file:
    for post in posts:
        print(post)
        reply = reply_by_chatgpt(post.strip())
        print(reply)
        output_file.write(reply + '\n')
