import os
import math

# ==========================================
# 1. CORE INTELLECTUAL BASES (CORE MEMORY)
# ==========================================
AI_DATA = {
    "greeting": (["hi", "hello", "hey"], "Hi there! I am your AI assistant."),
    "status": (["how are you"], "I am doing fantastic! What's about you?"),
    "identity": (["who are you", "about yourself"], "I am Falcon. I can solve math, science, and look up countries."),
    "physics_light": (["light", "reflection"], "Light is electromagnetic radiation. Angle of incidence equals angle of reflection."),
    "physics_universe": (["universe", "space"], "The universe contains all of space, time, matter, and energy stretching across billions of light-years."),
    "physics_velocity": (["velocity", "speed"], "Velocity defines the rate of change of an object's position with direction (v = d/t)."),
    "chemistry_elements": (["elements", "periodic"], "An element is a pure substance consisting of only one type of atom, classified by atomic number."),
    "chemistry_compounds": (["compounds", "molecules"], "A compound is a substance formed when two or more distinct chemical elements are chemically bonded."),
    "chemistry_reactions": (["reactions", "chemical"], "A chemical reaction rearranges constituent atoms of reactants to form new products."),
    "biology_respiration": (["respiration", "breathing"], "Respiration is the biological process where chemical energy from nutrients is released and converted into ATP."),
    "biology_cells": (["cells", "cell structure"], "Cells are the basic structural, functional, and biological building blocks of all living organisms."),
    "biology_life": (["life", "living things"], "Life is a characteristic driven by metabolism and reproduction that distinguishes organic matter.")
}

COUNTRIES = {
    "india": "New Delhi. PM: Narendra Modi.", "china": "Beijing. President: Xi Jinping.", 
    "usa": "Washington, D.C. President: Joe Biden.", "uk": "London. PM: Keir Starmer.", 
    "russia": "Moscow. President: Vladimir Putin.", "japan": "Tokyo. PM: Shigeru Ishiba.", 
    "germany": "Berlin. Chancellor: Olaf Scholz.", "france": "Paris. President: Emmanuel Macron.", 
    "canada": "Ottawa. PM: Justin Trudeau.", "australia": "Canberra. PM: Anthony Albanese.", 
    "italy": "Rome. PM: Giorgia Meloni.", "brazil": "Brasilia. President: Lula da Silva.", 
    "south africa": "Pretoria. President: Cyril Ramaphosa.", "south korea": "Seoul. President: Yoon Suk-yeol.", 
    "mexico": "Mexico City. President: Claudia Sheinbaum.", "saudi arabia": "Riyadh. PM: Crown Prince Mohammed bin Salman.", 
    "uae": "Abu Dhabi. President: Mohamed bin Zayed Al Nhayan.", "indonesia": "Jakarta. President: Prabowo Subianto.", 
    "turkey": "Ankara. President: Recep Tayyip Erdoğan.", "spain": "Madrid. PM: Pedro Sánchez.", 
    "singapore": "Singapore. PM: Lawrence Wong.", "netherlands": "Amsterdam. PM: Dick Schoof.", 
    "switzerland": "Bern. Managed by a Federal Council structure.", "sweden": "Stockholm. PM: Ulf Kristersson.", 
    "norway": "Oslo. PM: Jonas Gahr Støre.", "egypt": "Cairo. PM: Mostafa Madbouly.", 
    "pakistan": "Islamabad. PM: Shehbaz Sharif.", "bangladesh": "Dhaka. Chief Adviser: Muhammad Yunus.", 
    "sri lanka": "Sri Jayawardenepura Kotte. PM: Harini Amarasuriya.", "nepal": "Kathmandu. PM: K.P. Sharma Oli.", 
    "new zealand": "Wellington. PM: Christopher Luxon.", "argentina": "Buenos Aires. President: Javier Milei.", 
    "vietnam": "Hanoi. PM: Pham Minh Chinh.", "thailand": "Bangkok. PM: Paetongtarn Shinawatra.", 
    "malaysia": "Kuala Lumpur. PM: Anwar Ibrahim.", "philippines": "Manila. President: Ferdinand Marcos Jr.", 
    "nigeria": "Abuja. President: Bola Tinubu.", "kenya": "Nairobi. President: William Ruto.", 
    "israel": "Jerusalem. PM: Benjamin Netanyahu.", "iran": "Tehran. President: Masoud Pezeshkian.", 
    "iraq": "Baghdad. PM: Mohammed Shia' al-Sudani.", "ukraine": "Kyiv. President: Volodymyr Zelenskyy.", 
    "poland": "Warsaw. PM: Donald Tusk.", "belgium": "Brussels. PM: Alexander De Croo.", 
    "austria": "Vienna. Chancellor: Karl Nehammer.", "greece": "Athens. PM: Kyriakos Mitsotakis.", 
    "portugal": "Lisbon. PM: Luís Montenegro.", "colombia": "Bogota. President: Gustavo Petro.", 
    "chile": "Santiago. President: Gabriel Boric.", "peru": "Lima. PM: Gustavo Adrianzén."
}

INTERNAL_DIR = os.getcwd()
MEMORY_FILE = os.path.join(INTERNAL_DIR, "falcon_mem.txt")
LEARNED_MEMORY = {}

# ==========================================
# 2. SEPARATED STORAGE DRIVER ENGINE
# ==========================================
def load_memory():
    LEARNED_MEMORY.clear()
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if "::=" in line:
                        q, a = line.strip().split("::=", 1)
                        LEARNED_MEMORY[q.lower().strip()] = a.strip()
        except:
            pass

def rewrite_entire_file():
    """Completely rebuilds the file to clean out old entries upon rewrite corrections."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            for q, a in LEARNED_MEMORY.items():
                f.write(f"{q}::={a}\n")
    except:
        pass

def save_memory(q, a):
    clean_q = q.lower().strip()
    LEARNED_MEMORY[clean_q] = a.strip()
    rewrite_entire_file()

def tokenize(text):
    return "".join([c.lower() if (c.isalnum() or c.isspace() or c in "+-*/.") else " " for c in text]).split()

# ==========================================
# 3. ROUTED EXECUTOR CHANNELS
# ==========================================
def handle_math(msg):
    try:
        clean = "".join([c for c in msg if c in "0123456789+-*/.() "])
        if clean.strip() and any(op in clean for op in "+-*/"):
            return f"Result: {eval(clean)}"
    except:
        pass
    return "Math context found, but couldn't solve. Try clean formats like 5+5."

def handle_country(msg):
    for c, d in COUNTRIES.items():
        if c in msg.lower():
            return f"{c.title()} -> {d}"
    return "I know this is a country request, but that country isn't in my top 50 list."

def handle_science(msg):
    t = tokenize(msg)
    for k, (kw, resp) in AI_DATA.items():
        if any(w in t for w in kw) or any(w in msg.lower() for w in kw):
            return resp
    return "Science topic matched, but I don't have that specific card in my memory."

# ==========================================
# 4. NEURAL ROUTER MATRIX
# ==========================================
intent_features = {
    "math": ["calculate", "plus", "minus", "divide", "multiply", "solve", "math", "+", "-", "*", "/"],
    "country": ["capital", "prime", "minister", "pm", "president", "country", "who rules", "city"],
    "science": ["explain", "science", "light", "atom", "cells", "universe", "elements", "reaction", "physics", "biology", "chemistry", "hi", "hello", "hey"]
}

vocab = []
for words in intent_features.values():
    vocab.extend(words)
vocab = sorted(list(set(vocab)))
weights = {intnt: [1.0 if f in intent_features[intnt] else 0.0 for f in vocab] for intnt in intent_features}

def predict_intent(user_msg):
    m_low = user_msg.lower().strip()
    
    # Check core data overrides first to guarantee isolation rules stay strong
    for c in COUNTRIES:
        if c in m_low:
            return "country"
    for k, (kw, _) in AI_DATA.items():
        if any(w in m_low for w in kw):
            return "science"
            
    # Check dynamic tracking memory maps next
    if m_low in LEARNED_MEMORY:
        return "learned"
        
    t = tokenize(user_msg)
    vec = [1 if (f in t or f in m_low) else 0 for f in vocab]
    scores = {}
    for intnt in weights:
        dot = sum(vec[i] * weights[intnt][i] for i in range(len(vocab)))
        scores[intnt] = 1 / (1 + math.exp(-dot)) if dot > 0 else 0.0
        
    if max(scores.values()) == 0.0:
        return "math" if any(o in user_msg for o in "+-*/") else "science"
    return max(scores, key=scores.get)

# ==========================================
# 5. HARD TERMINAL CONTROLLER RUNNER LOOP
# ==========================================
load_memory()
print("========================================")
print("Falcon AI: Neural Routing Engine Online")
print("Status: 100% Offline | Memory Isolated: SAFE")
print("Type 'exit' to turn off the AI.")
print("========================================")

while True:
    inp = input("\nYou: ").strip()
    if inp.lower() == 'exit':
        print("Goodbye!")
        break
    if not inp:
        continue
    
    route = predict_intent(inp)
    
    # Custom interaction interception for editing old memories cleanly
    if route == "learned":
        print("[Neural Route -> ISOLATED LEARNED DRAWER]")
        print(f"Falcon: {LEARNED_MEMORY[inp.lower()]}")
        
        correct_check = input("Falcon: I already know this phrase. Overwrite/correct it? (yes/no): ").strip().lower()
        if correct_check in ['yes', 'y']:
            new_ans = input("Falcon: Enter the corrected response: ").strip()
            save_memory(inp, new_ans)
            print("Falcon: Memory successfully updated and rewritten permanently!")
        continue
        
    print(f"[Neural Route -> {route.upper()}]")
    resp = handle_math(inp) if route == "math" else (handle_country(inp) if route == "country" else handle_science(inp))
    
    if "couldn't solve" in resp or "isn't in my" in resp or "don't have that" in resp:
        print(f"Falcon: {resp}")
        teach = input("Falcon: Teach me this specific phrase? (yes/no): ").strip().lower()
        if teach in ['yes', 'y']:
            ans = input("Falcon: What should my exact response be?: ").strip()
            save_memory(inp, ans)
            print("Falcon: Saved to separate storage drawer securely!")
    else:
        print(f"Falcon: {resp}")
