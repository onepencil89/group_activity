import streamlit as st
import streamlit.components.v1 as components

if 'messages' not in st.session_state:
    st.session_state.messages = []

# 메인 페이지
st.title("메인 페이지")
st.write("메인 콘텐츠입니다.")

# HTML로 플로팅 챗봇 구현
chatbot_html = """
<style>
    .chatbot-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 350px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        z-index: 9999;
    }
    .chatbot-header {
        background: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 10px 10px 0 0;
        font-weight: bold;
    }
    .chatbot-body {
        height: 400px;
        overflow-y: auto;
        padding: 15px;
    }
    .chatbot-input {
        padding: 10px;
        border-top: 1px solid #ddd;
    }
    .chatbot-toggle {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: #4CAF50;
        color: white;
        font-size: 24px;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        z-index: 9999;
    }
</style>

<button class="chatbot-toggle" onclick="toggleChat()">💬</button>

<div id="chatbot" class="chatbot-container" style="display: none;">
    <div class="chatbot-header">
        💬 챗봇
        <button onclick="toggleChat()" style="float: right; background: none; border: none; color: white; cursor: pointer;">✕</button>
    </div>
    <div class="chatbot-body" id="chat-messages">
        <p>안녕하세요! 무엇을 도와드릴까요?</p>
    </div>
    <div class="chatbot-input">
        <input type="text" id="user-input" style="width: 80%; padding: 8px;">
        <button onclick="sendMessage()" style="width: 18%; padding: 8px;">전송</button>
    </div>
</div>

<script>
    function toggleChat() {
        var chatbot = document.getElementById('chatbot');
        if (chatbot.style.display === 'none') {
            chatbot.style.display = 'block';
        } else {
            chatbot.style.display = 'none';
        }
    }
    
    function sendMessage() {
        var input = document.getElementById('user-input');
        var message = input.value;
        if (message.trim() !== '') {
            var chatMessages = document.getElementById('chat-messages');
            chatMessages.innerHTML += '<p><strong>나:</strong> ' + message + '</p>';
            chatMessages.innerHTML += '<p><strong>봇:</strong> ' + message + '에 대한 응답입니다.</p>';
            input.value = '';
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
</script>
"""

components.html(chatbot_html, height=600)