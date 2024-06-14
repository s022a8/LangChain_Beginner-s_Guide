import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="準備ができました！メッセージを入力してください！").send()

@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ: " + str(input_message))
    await cl.Message(content="こんにちは！").send()
    