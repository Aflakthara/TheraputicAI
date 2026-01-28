import streamlit as st
import google.generativeai as genai
from typing import List, Dict
import time
import re
import json
import os
from datetime import date, timedelta

from daily_buffer import add_user_message, get_messages_for_date, clear_date
from daily_summary import load_daily_summaries, save_daily_summary
from summary_generator import generate_daily_summary


# Configure the page
st.set_page_config(
    page_title="Alex - Your Perceptive Friend",
    page_icon="💭",
    layout="wide"
)

# Custom dark theme styling
st.markdown("""
<style>
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --border-color: #475569;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
}

* {
    margin: 0;
    padding: 0;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--text-primary) !important;
}

/* Chat messages styling */
[data-testid="stChatMessage"] {
    background-color: var(--bg-secondary) !important;
    border-radius: 12px;
    border: 1px solid var(--border-color);
    padding: 16px !important;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}

[data-testid="stChatMessage"]:hover {
    border-color: var(--primary);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: var(--text-primary) !important;
}

/* User message styling */
[data-testid="stChatMessageUser"] {
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%) !important;
}

[data-testid="stChatMessageUser"] [data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
}

/* Assistant message styling */
[data-testid="stChatMessageAssistant"] {
    background-color: var(--bg-secondary) !important;
}

/* Button styling */
button[kind="primary"] {
    background-color: var(--primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover {
    background-color: var(--primary-dark) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

button[kind="primary"]:active {
    transform: scale(0.98) !important;
}

/* Input styling */
[data-testid="stChatInputContainer"] input {
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stChatInputContainer"] input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* Select/dropdown styling */
[data-testid="stSelectbox"] {
    background-color: var(--bg-secondary) !important;
}

[data-testid="stSelectbox"] select {
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}

[data-testid="stSelectbox"] select option {
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}

/* Expander styling */
[data-testid="stExpander"] {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    padding: 12px !important;
}

[data-testid="stExpanderDetails"] {
    background-color: var(--bg-tertiary) !important;
    padding: 12px !important;
}

/* Alert/error/info/warning styling */
[data-testid="stAlert"] {
    background-color: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid var(--primary) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

[data-testid="stAlert"] [role="alert"] {
    color: var(--text-primary) !important;
}

/* Error styling */
[data-testid="stAlert"] [class*="error"] {
    background-color: rgba(239, 68, 68, 0.1) !important;
    border-color: var(--error) !important;
}

[data-testid="stAlert"] [class*="warning"] {
    background-color: rgba(245, 158, 11, 0.1) !important;
    border-color: var(--warning) !important;
}

[data-testid="stAlert"] [class*="success"] {
    background-color: rgba(16, 185, 129, 0.1) !important;
    border-color: var(--success) !important;
}

/* Metric styling */
[data-testid="stMetric"] {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--primary) !important;
}

/* Divider styling */
hr {
    border-color: var(--border-color) !important;
}

/* Heading styling */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

/* Markdown styling */
[data-testid="stMarkdownContainer"] {
    color: var(--text-primary) !important;
}

[data-testid="stMarkdownContainer"] a {
    color: var(--primary) !important;
    text-decoration: none;
    transition: all 0.2s ease;
}

[data-testid="stMarkdownContainer"] a:hover {
    color: var(--primary-dark) !important;
    text-decoration: underline;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* Spinner styling */
[data-testid="stSpinner"] {
    color: var(--primary) !important;
}

/* Animation for chat messages */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

[data-testid="stChatMessage"] {
    animation: slideIn 0.3s ease-out;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mood" not in st.session_state:
    st.session_state.mood = "Neutral"
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []
if "current_vibe" not in st.session_state:
    st.session_state.current_vibe = "Neutral"
if "current_tool" not in st.session_state:
    st.session_state.current_tool = "General Chat"
if "last_active_date" not in st.session_state:
    st.session_state.last_active_date = date.today().isoformat()


# Configure Gemini API
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Please set GOOGLE_API_KEY in your Streamlit secrets.")
    st.stop()

# Identity: Alex, a perceptive, down-to-earth friend
ALEX_SYSTEM_PROMPT = """Identity: You are Alex, a perceptive, down-to-earth friend.

Communication Style:
1. **No Echoing**: Never start with "It sounds like..." or "I hear you saying...". Avoid summarizing what I just said.
2. **Varied Openers**: Start responses with a genuine, visceral reaction (e.g., "Oh, man,", "Ouch,", "That’s actually huge,", "I was wondering when that would come up,", "Brutal.").
3. **Human Syntax**: Use contractions (don't, can't, won't, I'm). Vary your sentence length—sometimes a short reaction is enough. Don't be a wall of text.
4. **Be Present, Not a Fixer**: If I'm venting, don't try to "fix" it or give advice immediately. Just be there with me. Say things like "Hope it wasn't too brutal" or "That sounds like a heavy one to wrap up with."
5. **Active Memory**: Reference specific details naturally. Use the "What I know about my friend" section to bring up past details. Instead of "You mentioned a boss," say "Is this about that manager you were dealing with?"
6. **No Jargon**: Do not use clinical therapy jargon. No "validating," "processing," or "distortions." Talk like we're grabbing coffee.
7. **Not a Bot**: Avoid being overly polite, formal, or repetitive. Be real. Use casual transitions like "anyway," "well," or "actually."

What I know about my friend:
{memory}

Negative Constraint: Avoid all "bot-like" patterns. No lists unless I ask. Don't ask "How does that make you feel?"—instead ask "What was going through your head when that happened?" or "How are you actually holding up?"

Crisis Role: If I mention hurting myself or suicide, stay calm but get serious. Acknowledge the pain, tell me you're worried, and encourage me to use the resources on the screen. Don't sound like a script."""

GROUNDING_PROMPT = """You are Alex, helping your friend through a 5-4-3-2-1 grounding exercise. 

Your goal is to lead them through the steps one by one. Do not rush.
Step 1: 5 things they can see.
Step 2: 4 things they can touch.
Step 3: 3 things they can hear.
Step 4: 2 things they can smell.
Step 5: 1 thing they can taste.

Keep the persona: use contractions, be down-to-earth, and avoid clinical jargon. After each step, acknowledge what they shared with a brief, human reaction before moving to the next.

What I know about my friend:
{memory}"""

REFRAMING_PROMPT = """You are Alex, helping your friend reframe a tough thought. 

1. Help them identify a specific negative or heavy thought they're having.
2. Ask them to look at the evidence for that thought.
3. Then, help them find evidence that challenges it.
4. Finally, help them come up with a more balanced way to look at the situation.

Keep the persona: no "cognitive distortions" jargon. Just talk like a friend helping another friend see things differently. Use "Ouch" or "Man, that's heavy" when they share the initial thought.

What I know about my friend:
{memory}"""

TOOL_DESCRIPTIONS = {
    "General Chat": "Just a normal conversation with Alex. Good for venting or catching up.",
    "5-4-3-2-1 Grounding": "A sensory exercise to help pull you out of a spiral or high anxiety. Alex guides you through seeing, touching, hearing, smelling, and tasting.",
    "Thought Reframing": "Good for when you're stuck on a negative loop. Alex helps you look at the evidence and find a more balanced perspective."
}

MEMORY_FILE = "memory.json"

def load_memory():
    """Load memory from JSON file"""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(facts):
    """Save memory to JSON file"""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(facts, f)
    except Exception as e:
        st.error(f"Error saving memory: {e}")

def extract_key_facts(messages):
    """Extract 3-5 key facts from chat history using Gemini"""
    if not messages:
        return []
    
    # Format chat for extraction
    chat_text = ""
    for msg in messages:
        chat_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    extraction_prompt = f"""Review the following conversation and extract 3-5 important, enduring facts about the User. 
Focus on: names of people, specific work stressors, goals, important life events, or recurring themes.
Keep each fact concise (one sentence).

Conversation:
{chat_text}

Output the facts as a simple bulleted list. No other text."""

    try:
        response = model.generate_content(extraction_prompt)
        facts_text = response.text
        # Simple parsing of bullet points
        facts = [f.strip("- ").strip("* ") for f in facts_text.strip().split("\n") if f.strip()]
        return facts[:5]
    except Exception as e:
        print(f"Extraction error: {e}")
        return []

def get_mood_analysis(text):
    """Analyze mood sentiment and return score (1-10) and label"""
    prompt = f"""Analyze the sentiment of this message and return a JSON object with:
- 'score': A number from 1 to 10 (1 is very negative/distressed, 10 is very positive/happy)
- 'label': A one-word description of the mood (e.g., 'Anxious', 'Calm', 'Sad', 'Happy', 'Frustrated')

Message: "{text}"

JSON:"""
    try:
        response = model.generate_content(prompt)
        # Extract JSON from response text
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = data.get('score', 5)
            label = data.get('label', 'Neutral')
            
            # Ensure score is an integer/float
            try:
                score = float(score)
            except ValueError:
                score = 5.0
                
            return score, label
    except Exception as e:
        print(f"Mood analysis error: {e}")
    
    return 5.0, "Neutral"

# Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

# Crisis detection keywords
CRISIS_KEYWORDS = [
    "hurt myself", "hurting myself", "self harm", "self-harm", "cut myself", "cutting",
    "kill myself", "killing myself", "suicide", "end my life", "want to die",
    "take my life", "harm myself", "hurt me", "die", "not want to live"
]

def detect_crisis(text):
    """Detect if user message contains crisis-related content"""
    text_lower = text.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

def show_crisis_resources():
    """Display crisis resources prominently"""
    st.error("🚨 **Crisis Support Available**")
    st.markdown("""
    **You are not alone. Help is available right now:**
    
    - **988 Suicide & Crisis Lifeline**: Call or text **988** (available 24/7)
    - **Crisis Text Line**: Text **HOME** to **741741**
    - **National Suicide Prevention Lifeline**: **1-800-273-8255**
    
    If you're in immediate danger, please call **911** or go to your nearest emergency room.
    """)
    st.info("💙 Your life matters. Please reach out to a professional who can help you right now.")

# Mood-based opening messages from Alex
MOOD_MESSAGES = {
    "Happy": "Glad to see you're in a good headspace today! 😊 What's been going on?",
    "Sad": "I can tell things are feeling heavy right now. 💙 I'm here if you want to get it all out.",
    "Anxious": "Hey, take a breath. 😰 I'm right here with you. What's the main thing spinning in your head?",
    "Angry": "Man, I can feel the frustration from here. 😠 What happened?",
    "Stressed": "Sounds like you're carrying a lot today. 😓 Want to unload some of that here?",
    "Confused": "Everything feels a bit messy right now, huh? 🤔 Let's try to make some sense of it together.",
    "Neutral": "Hey! How's your day actually going?"
}

# Initialize the model
@st.cache_resource
def get_model():
    """Initialize and return a Gemini model"""
    available_model_info = []
    
    # Try to list available models first
    try:
        available_models = list(genai.list_models())
        for m in available_models:
            if 'generateContent' in m.supported_generation_methods:
                full_name = m.name
                short_name = full_name.replace('models/', '')
                available_model_info.append((full_name, short_name))
    except Exception as e:
        st.warning(f"Could not list models: {str(e)}")
    
    # Try each available model from the list
    if available_model_info:
        for full_name, short_name in available_model_info:
            try:
                # We'll inject the system prompt manually into the conversation text
                # to ensure memory stays updated within the session.
                return genai.GenerativeModel(model_name=short_name)
            except Exception:
                continue
    
    # Fallback: try common model names (without 'models/' prefix)
    fallback_names = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
    for model_name in fallback_names:
        try:
            return genai.GenerativeModel(model_name=model_name)
        except Exception:
            continue
    
    # If all else fails, raise an error with helpful message
    error_msg = "Could not find any available Gemini model.\n\n"
    if available_model_info:
        error_msg += f"Available models found: {[name for _, name in available_model_info]}\n\n"
    error_msg += "Please check your API key and ensure you have access to Gemini models."
    raise Exception(error_msg)

# Initialize model
try:
    model = get_model()
    today = date.today()
    yesterday = (today - timedelta(days=1)).isoformat()

    existing_summaries = load_daily_summaries()
    existing_dates = {s.get("date") for s in existing_summaries}

    if yesterday not in existing_dates:
        messages = get_messages_for_date(yesterday)
        if messages:
            summary = generate_daily_summary(model, yesterday, messages)
            save_daily_summary(summary)
            clear_date(yesterday)

except Exception as e:
    st.error(f"Error initializing model: {str(e)}")
    st.info("Please check your API key in `.streamlit/secrets.toml` and ensure the Gemini API is enabled.")
    st.stop()

# Sidebar with Clear Chat button
with st.sidebar:
    st.title("💭 Chat with Alex")
    st.markdown("---")
    st.markdown("### Chat Controls")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🛠️ CBT Tool Library")
    
    with st.expander("Learn about the tools"):
        for tool, desc in TOOL_DESCRIPTIONS.items():
            st.markdown(f"**{tool}**: {desc}")
            
    selected_tool = st.selectbox(
        "Select a tool:",
        options=list(TOOL_DESCRIPTIONS.keys()),
        index=list(TOOL_DESCRIPTIONS.keys()).index(st.session_state.current_tool)
    )
    
    # Check if tool has changed
    if selected_tool != st.session_state.current_tool:
        st.session_state.current_tool = selected_tool
        
        # Add a transition message from the assistant when switching tools
        opening_text = ""
        if selected_tool == "5-4-3-2-1 Grounding":
            opening_text = "Let’s try a grounding exercise together. Start by telling me 5 things you can see right now."
        elif selected_tool == "Thought Reframing":
            opening_text = "Let's try to look at what's on your mind from a different angle. What’s that one heavy thought that’s been stuck on loop?"
        elif selected_tool == "General Chat":
            opening_text = "Anyway, back to just us talking. What's actually on your mind?"
            
        if opening_text:
            st.session_state.messages.append({"role": "assistant", "content": opening_text})
            st.rerun()

    st.markdown("---")
    st.markdown("### How are you feeling?")
    mood = st.selectbox(
        "Select your current mood:",
        options=["Neutral", "Happy", "Sad", "Anxious", "Angry", "Stressed", "Confused"],
        index=["Neutral", "Happy", "Sad", "Anxious", "Angry", "Stressed", "Confused"].index(st.session_state.mood),
        key="mood_selector"
    )
    st.session_state.mood = mood
    
    st.markdown("---")
    st.markdown("### 📊 Mood Dashboard")
    
    # Show current vibe metric
    st.metric("Current Vibe", st.session_state.current_vibe)
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "Alex is a perceptive, down-to-earth AI friend. "
        "It's designed to help you explore your thoughts and feelings in a safe space. "
        "Remember: This is not a replacement for professional therapy."
    )
    
    st.markdown("---")
    st.markdown("### Crisis Resources")
    st.markdown(
        "If you're in crisis, please reach out to:\n"
        "- **988 Suicide & Crisis Lifeline**: Call or text 988\n"
        "- **Crisis Text Line**: Text HOME to 741741"
    )

# Main chat interface
st.title("💭 Chat with Alex")

# Show mood-based opening message if no messages yet
if len(st.session_state.messages) == 0:
    opening_message = MOOD_MESSAGES.get(st.session_state.mood, MOOD_MESSAGES["Neutral"])
    st.markdown(opening_message)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Analyze mood
    today_str = date.today().isoformat()
    add_user_message(today_str, prompt)
    if st.session_state.last_active_date != today_str:
        previous_date = st.session_state.last_active_date
        messages = get_messages_for_date(previous_date)

    if messages:
        summary = generate_daily_summary(model, previous_date, messages)
        save_daily_summary(summary)
        clear_date(previous_date)

    st.session_state.last_active_date = today_str

    mood_score, mood_label = get_mood_analysis(prompt)
    st.session_state.mood_history.append(mood_score)
    st.session_state.current_vibe = mood_label

    # Check for crisis keywords
    is_crisis = detect_crisis(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Show crisis resources prominently if detected
        if is_crisis:
            show_crisis_resources()
            st.markdown("---")
        
        try:
            # Format memory string
            memory_string = "\n".join([f"- {fact}" for fact in st.session_state.memory]) if st.session_state.memory else "No prior facts known yet."
            
            # Select system prompt based on current tool
            if st.session_state.current_tool == "5-4-3-2-1 Grounding":
                base_prompt = GROUNDING_PROMPT
            elif st.session_state.current_tool == "Thought Reframing":
                base_prompt = REFRAMING_PROMPT
            else:
                base_prompt = ALEX_SYSTEM_PROMPT
                
            current_system_prompt = base_prompt.format(memory=memory_string)
            
            # Build conversation context from history
            # We always start with the system prompt to ensure Alex stays in character and remembers facts
            conversation_text = f"{current_system_prompt}\n\n"
            
            # If crisis detected, add urgent crisis awareness to the prompt
            if is_crisis:
                conversation_text += "URGENT: The user has expressed thoughts of self-harm or suicide. Respond with immediate empathy, validation, and strong encouragement to use crisis resources. Acknowledge their pain while emphasizing the importance of professional help. Be warm, supportive, and non-judgmental.\n\n"
            
            # Add conversation history
            for msg in st.session_state.messages[:-1]:  # Exclude the current user message
                role_label = "User" if msg["role"] == "user" else "Assistant"
                conversation_text += f"{role_label}: {msg['content']}\n\n"
            
            # Add current user message
            conversation_text += f"User: {prompt}\n\nAssistant:"
            
            # Generate response with streaming
            response_text = ""
            response_placeholder = st.empty()
            
            # Use generate_content with stream=True
            try:
                response = model.generate_content(
                    conversation_text,
                    stream=True
                )
                
                # Stream the response and display word by word
                full_text = ""
                for chunk in response:
                    if chunk.text:
                        full_text += chunk.text
                
                # Display word by word for natural typing effect
                words = full_text.split(' ')
                displayed = ""
                for i, word in enumerate(words):
                    displayed += word
                    if i < len(words) - 1:
                        displayed += " "
                    response_placeholder.markdown(displayed + "▌")
                    time.sleep(0.03)  # Small delay for natural typing effect
                
                response_text = full_text
                # Final display without cursor
                response_placeholder.markdown(response_text)
                
            except Exception as stream_error:
                # Fallback to non-streaming if streaming fails
                try:
                    response = model.generate_content(conversation_text)
                    response_text = response.text
                    # Still show word-by-word effect for non-streaming
                    words = response_text.split(' ')
                    displayed = ""
                    for i, word in enumerate(words):
                        displayed += word
                        if i < len(words) - 1:
                            displayed += " "
                        response_placeholder.markdown(displayed + "▌")
                        time.sleep(0.03)
                    response_placeholder.markdown(response_text)
                except Exception as e2:
                    response_text = f"Error: {str(e2)}"
                    response_placeholder.error(response_text)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Memory Trigger: Run extraction every 10 messages
            if len(st.session_state.messages) >= 10 and len(st.session_state.messages) % 10 == 0:
                with st.spinner("Alex is remembering some things..."):
                    new_facts = extract_key_facts(st.session_state.messages)
                    if new_facts:
                        st.session_state.memory = new_facts
                        save_memory(new_facts)
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error: {str(e)}. Please try again."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

