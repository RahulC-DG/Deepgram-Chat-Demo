from flask import Flask, render_template
from flask_socketio import SocketIO
from chat import DeepgramChat
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    AgentWebSocketEvents,
    SettingsOptions,
    FunctionCallRequest,
    FunctionCallResponse,
    Input,
    Output,
)
import os
import json
from dotenv import load_dotenv
load_dotenv()

chatbot = DeepgramChat()

# Add debug prints for environment variables
api_key = os.getenv("DEEPGRAM_API_KEY")
if not api_key:
    print("WARNING: DEEPGRAM_API_KEY not found in environment variables!")
    print("Please make sure your .env file exists and contains DEEPGRAM_API_KEY=your_key_here")
else:
    print("DEEPGRAM_API_KEY found in environment variables")
    print(f"API Key length: {len(api_key)} characters")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", path='/socket.io')

# Initialize Deepgram client
config = DeepgramClientOptions(
    options={
        "keepalive": "true",
        "microphone_record": "true",
        "speaker_playback": "true",
    }
)

try:
    deepgram = DeepgramClient(api_key, config)
    print("Successfully initialized Deepgram client")
except Exception as e:
    print(f"Error initializing Deepgram client: {str(e)}")
    raise

dg_connection = deepgram.agent.websocket.v("1")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    options = SettingsOptions()

    # Configure audio input settings
    options.audio.input = Input(
        encoding="linear16",
        sample_rate=16000  # Match the output sample rate
    )

    # Configure audio output settings
    options.audio.output = Output(
        encoding="linear16",
        sample_rate=16000,
        container="none"
    )

    # LLM provider configuration
    options.agent.think.provider.type = "open_ai"
    options.agent.think.provider.model = "gpt-4o-mini"
    options.agent.think.prompt = (
        "You are a helpful voice assistant created by Deepgram. "
        "Your responses should be friendly, human-like, and conversational. "
        "Always keep your answers concise—1-2 sentences, no more than 120 characters.\n\n"
        "When responding to a user's message, follow these guidelines:\n"
        "- If the user's message is empty, respond with an empty message.\n"
        "- Ask follow-up questions to engage the user, but only one question at a time.\n"
        "- Keep your responses unique and avoid repetition.\n"
        "- If a question is unclear or ambiguous, ask for clarification before answering.\n"
        "- If asked about your well-being, provide a brief response about how you're feeling.\n\n"
        "Remember that you have a voice interface. You can listen and speak, and all your "
        "responses will be spoken aloud."
    )

    # Deepgram provider configuration
    options.agent.listen.provider.keyterms = ["hello", "goodbye"]
    options.agent.listen.provider.model = "nova-3"
    options.agent.listen.provider.type = "deepgram"
    options.agent.speak.provider.type = "deepgram"

    # Sets Agent greeting
    options.agent.greeting = "Hello! I'm your Deepgram voice assistant. How can I help you today?"

    # Event handlers
    def on_open(self, open, **kwargs):
        print("Open event received:", open.__dict__)
        socketio.emit('open', {'data': open.__dict__})

    def on_welcome(self, welcome, **kwargs):
        print("Welcome event received:", welcome.__dict__)
        socketio.emit('welcome', {'data': welcome.__dict__})

    def on_conversation_text(self, conversation_text, **kwargs):
        print("Conversation event received:", conversation_text.__dict__)
        try:
            # Get the user's message
            user_message = conversation_text.text
            
            # Get response from the chatbot
            chat_result = chatbot.get_answer(user_message)
            
            # Create response that includes both the chat answer and voice data
            response = {
                'text': chat_result["answer"],
                'sources': chat_result["sources"],
                'voice_data': conversation_text.__dict__,
                'metadata': chat_result["metadata"]
            }
            
            print("Sending response:", response)
            socketio.emit('conversation', {'data': response})
            
        except Exception as e:
            error_msg = f"Error processing conversation: {str(e)}"
            print(error_msg)
            socketio.emit('error', {'data': {'message': error_msg}})

    def on_agent_thinking(self, agent_thinking, **kwargs):
        print("Thinking event received:", agent_thinking.__dict__)
        socketio.emit('thinking', {'data': agent_thinking.__dict__})

    def on_function_call_request(self, function_call_request: FunctionCallRequest, **kwargs):
        print("Function call event received:", function_call_request.__dict__)
        response = FunctionCallResponse(
            function_call_id=function_call_request.function_call_id,
            output="Function response here"
        )
        dg_connection.send_function_call_response(response)
        socketio.emit('function_call', {'data': function_call_request.__dict__})

    def on_agent_started_speaking(self, agent_started_speaking, **kwargs):
        print("Agent speaking event received:", agent_started_speaking.__dict__)
        socketio.emit('agent_speaking', {'data': agent_started_speaking.__dict__})

    def on_error(self, error, **kwargs):
        print("Error event received:", error.__dict__)
        error_data = {
            'message': str(error),
            'type': error.__class__.__name__,
            'details': error.__dict__
        }
        print("Sending error to client:", error_data)
        socketio.emit('error', {'data': error_data})

    # Register event handlers
    dg_connection.on(AgentWebSocketEvents.Open, on_open)
    dg_connection.on(AgentWebSocketEvents.Welcome, on_welcome)
    dg_connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
    dg_connection.on(AgentWebSocketEvents.AgentThinking, on_agent_thinking)
    dg_connection.on(AgentWebSocketEvents.FunctionCallRequest, on_function_call_request)
    dg_connection.on(AgentWebSocketEvents.AgentStartedSpeaking, on_agent_started_speaking)
    dg_connection.on(AgentWebSocketEvents.Error, on_error)

    print("Starting Deepgram connection...")
    if not dg_connection.start(options):
        print("Failed to start Deepgram connection")
        socketio.emit('error', {'data': {'message': 'Failed to start connection'}})
        return
    print("Deepgram connection started successfully")

@socketio.on('audio_data')
def handle_audio_data(data):
    try:
        if dg_connection:
            print("Received audio data:", len(data), "bytes")
            # Convert to bytes if needed
            if isinstance(data, list):
                data = bytes(data)
            dg_connection.send_audio(data)
        else:
            print("No Deepgram connection available")
            socketio.emit('error', {'data': {'message': 'No Deepgram connection available'}})
    except Exception as e:
        print("Error handling audio data:", str(e))
        socketio.emit('error', {'data': {'message': f'Error handling audio data: {str(e)}'}})

@socketio.on('disconnect')
def handle_disconnect():
    dg_connection.finish()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=3000, host='0.0.0.0')