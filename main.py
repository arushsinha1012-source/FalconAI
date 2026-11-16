import asyncio
import math
import random
import re
import flet as ft

class FalconNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        random.seed(42)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.w2 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.lr = 0.25

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))

    def sigmoid_deriv(self, x):
        return x * (1.0 - x)

    def forward(self, inputs):
        self.hidden = [
            self.sigmoid(sum(inputs[i] * self.w1[i][j] for i in range(self.input_size)))
            for j in range(self.hidden_size)
        ]
        self.output = [
            self.sigmoid(sum(self.hidden[j] * self.w2[j][k] for j in range(self.hidden_size)))
            for k in range(self.output_size)
        ]
        return self.output

    def train(self, data, epochs=1200):
        for _ in range(epochs):
            for inputs, targets in data:
                outputs = self.forward(inputs)
                out_errors = [targets[k] - outputs[k] for k in range(self.output_size)]
                out_deltas = [out_errors[k] * self.sigmoid_deriv(outputs[k]) for k in range(self.output_size)]
                hid_errors = [
                    sum(out_deltas[k] * self.w2[j][k] for k in range(self.output_size))
                    for j in range(self.hidden_size)
                ]
                hid_deltas = [hid_errors[j] * self.sigmoid_deriv(self.hidden[j]) for j in range(self.hidden_size)]

                for j in range(self.hidden_size):
                    for k in range(self.output_size):
                        self.w2[j][k] += self.lr * out_deltas[k] * self.hidden[j]
                for i in range(self.input_size):
                    for j in range(self.hidden_size):
                        self.w1[i][j] += self.lr * hid_deltas[j] * inputs[i]

VOCAB = [
    "hi", "hello", "hey", "how", "are", "you",
    "calculate", "solve", "math", "capital", "country",
    "physics", "chemistry", "biology", "atom", "cell",
    "force", "gravity", "dna", "acid"
]

def text_to_vector(text):
    words = re.findall(r'\w+', text.lower())
    return [1.0 if word in words else 0.0 for word in VOCAB]

TRAINING_DATA = [
    (text_to_vector("hi hello hey"), [1.0, 0.0, 0.0, 0.0, 0.0]),
    (text_to_vector("how are you"), [0.0, 1.0, 0.0, 0.0, 0.0]),
    (text_to_vector("calculate solve math"), [0.0, 0.0, 1.0, 0.0, 0.0]),
    (text_to_vector("what is the capital of country"), [0.0, 0.0, 0.0, 1.0, 0.0]),
    (text_to_vector("physics chemistry biology atom cell force gravity dna acid"), [0.0, 0.0, 0.0, 0.0, 1.0]),
]

ANN = FalconNeuralNetwork(len(VOCAB), 10, 5)
ANN.train(TRAINING_DATA)

CAPITALS = {
    "india": "New Delhi", "united states": "Washington, D.C.", "usa": "Washington, D.C.",
    "china": "Beijing", "japan": "Tokyo", "germany": "Berlin", "united kingdom": "London",
    "uk": "London", "france": "Paris", "italy": "Rome", "canada": "Ottawa",
    "south korea": "Seoul", "russia": "Moscow", "brazil": "Brasilia", "australia": "Canberra",
    "spain": "Madrid", "mexico": "Mexico City", "indonesia": "Jakarta", "netherlands": "Amsterdam",
    "saudi arabia": "Riyadh", "turkey": "Ankara", "switzerland": "Bern", "poland": "Warsaw",
    "sweden": "Stockholm", "belgium": "Brussels", "thailand": "Bangkok", "argentina": "Buenos Aires",
    "norway": "Oslo", "austria": "Vienna", "uae": "Abu Dhabi", "united arab emirates": "Abu Dhabi",
    "israel": "Jerusalem", "south africa": "Pretoria", "denmark": "Copenhagen", "singapore": "Singapore",
    "malaysia": "Kuala Lumpur", "philippines": "Manila", "egypt": "Cairo", "vietnam": "Hanoi",
    "bangladesh": "Dhaka", "pakistan": "Islamabad", "chile": "Santiago", "finland": "Helsinki",
    "colombia": "Bogota", "romania": "Bucharest", "czech republic": "Prague", "new zealand": "Wellington",
    "portugal": "Lisbon", "greece": "Athens", "peru": "Lima", "ukraine": "Kyiv", "ireland": "Dublin"
}

SCIENCE_DB = {
    "gravity": "Physics: Gravity is the universal force of attraction acting between all matter. On Earth, g = 9.8 m/s².",
    "force": "Physics: Force is a push or pull upon an object resulting from interaction with another object (F = m * a).",
    "newton": "Physics: 1st Law (Inertia), 2nd Law (F = ma), 3rd Law (Every action has an equal and opposite reaction).",
    "light": "Physics: Light is electromagnetic radiation traveling at approximately 299,792,458 m/s in a vacuum.",
    "energy": "Physics: Energy is the capacity to do work and cannot be created or destroyed (Law of Conservation of Energy).",
    "relativity": "Physics: Einstein's theory linking space, time, and mass-energy equivalence (E = mc²).",
    "atom": "Chemistry: An atom is the basic unit of matter with a nucleus (protons + neutrons) surrounded by electrons.",
    "periodic table": "Chemistry: Organizes all 118 known chemical elements by increasing atomic number.",
    "acid": "Chemistry: A substance that donates hydrogen ions (H+) in water and has a pH less than 7.",
    "base": "Chemistry: A substance that accepts H+ ions or releases OH- ions and has a pH greater than 7.",
    "water": "Chemistry: Water (H2O) is a polar covalent molecule of two hydrogen atoms and one oxygen atom.",
    "photosynthesis": "Biology & Chemistry: Green plants convert sunlight, water, and CO2 into glucose and oxygen.",
    "cell": "Biology: The cell is the basic structural and functional unit of all living organisms.",
    "dna": "Biology: Deoxyribonucleic acid (DNA) carries the genetic instructions for all living organisms.",
    "mitochondria": "Biology: The powerhouse of the cell that generates ATP energy.",
    "evolution": "Biology: Change in heritable traits of populations over generations via natural selection.",
    "heart": "Biology: A four-chambered muscular organ that pumps blood through the circulatory system."
}

def safe_math_eval(expr):
    cleaned = re.sub(r'[^0-9+\-*/().%^ ]', '', expr).replace('^', '**')
    if not any(ch.isdigit() for ch in cleaned):
        return None
    if not any(op in cleaned for op in ['+', '-', '*', '/', '%', '**']):
        return None
    try:
        result = eval(cleaned, {"__builtins__": None}, {})
        return f"Calculation Result: {result}"
    except Exception:
        return None

def get_falcon_response(user_input):
    raw = user_input.strip()
    low = raw.lower().strip("?!., ")

    if low in ["hi", "hello", "hey"]:
        return "Hi,I am Falcon"
    if low in ["how are you", "how are you doing", "how r u"]:
        return "I am doing fantastic!"

    math_res = safe_math_eval(raw)
    if math_res is not None:
        return math_res

    for country, capital in CAPITALS.items():
        if country in low:
            return f"The capital of {country.title()} is {capital}."

    for topic, explanation in SCIENCE_DB.items():
        if topic in low:
            return explanation

    vec = text_to_vector(low)
    prediction = ANN.forward(vec)
    intent_idx = prediction.index(max(prediction))

    if intent_idx == 0:
        return "Hi,I am Falcon"
    elif intent_idx == 1:
        return "I am doing fantastic!"
    elif intent_idx == 2:
        return "Please enter a valid math equation (e.g., 25 * 4)."
    elif intent_idx == 3:
        return "That country is not in my top 50 offline database."
    elif intent_idx == 4:
        return "That topic is not in my offline database. Ask about gravity, force, atom, cell, DNA, or acid."

    return "Sorry, I can only answer questions available in my offline database."

async def main(page: ft.Page):
    page.title = "Falcon AI"
    page.theme_mode = "light"
    page.bgcolor = "#ffffff"
    page.padding = 0

    def show_main_screen():
        page.controls.clear()
        page.bgcolor = "#ffffff"
        page.padding = 20

        welcome_title = ft.Container(
            content=ft.Text(
                "Welcome To Falcon!",
                size=26,
                weight="bold",
                color="#000000",
                text_align="center"
            ),
            alignment=ft.alignment.center,
            padding=ft.padding.only(top=20, bottom=10)
        )

        chat_list = ft.ListView(expand=True, spacing=12, auto_scroll=True)
        input_box = ft.TextField(
            hint_text="Ask Falcon...",
            expand=True,
            border_radius=10,
            color="#000000",
            bgcolor="#f8fafc",
            border_color="#cbd5e1"
        )

        def send_click(e):
            msg = input_box.value.strip()
            if not msg:
                return
            chat_list.controls.append(ft.Text(f"You: {msg}", color="#1e293b", size=15))
            input_box.value = ""
            reply = get_falcon_response(msg)
            chat_list.controls.append(ft.Text(f"Falcon: {reply}", color="#0f172a", weight="bold", size=15))
            page.update()

        input_box.on_submit = send_click
        send_btn = ft.ElevatedButton("Send", on_click=send_click, bgcolor="#0f172a", color="#ffffff")

        page.add(welcome_title, chat_list, ft.Row([input_box, send_btn], alignment="center"))
        page.update()

    already_opened = await page.client_storage.get_async("falcon_first_launch_done")

    if not already_opened:
        await page.client_storage.set_async("falcon_first_launch_done", True)
        page.bgcolor = "#000000"
        intro_active = True

        def finish_intro():
            nonlocal intro_active
            if not intro_active:
                return
            intro_active = False
            try:
                bg_audio.pause()
            except Exception:
                pass
            page.overlay.clear()
            show_main_screen()

