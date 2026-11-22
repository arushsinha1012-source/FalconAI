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
    "force", "gravity", "dna", "acid", "energy", "planet"
]


def text_to_vector(text):
    words = re.findall(r'\w+', text.lower())
    return [1.0 if word in words else 0.0 for word in VOCAB]


TRAINING_DATA = [
    (text_to_vector("hi hello hey"), [1.0, 0.0, 0.0, 0.0, 0.0]),
    (text_to_vector("how are you"), [0.0, 1.0, 0.0, 0.0, 0.0]),
    (text_to_vector("calculate solve math"), [0.0, 0.0, 1.0, 0.0, 0.0]),
    (text_to_vector("what is the capital of country"), [0.0, 0.0, 0.0, 1.0, 0.0]),
    (text_to_vector("physics chemistry biology atom cell force gravity dna acid energy planet"), [0.0, 0.0, 0.0, 0.0, 1.0]),
]

ANN = FalconNeuralNetwork(len(VOCAB), 12, 5)
ANN.train(TRAINING_DATA)

CAPITALS = {
    "afghanistan": "Kabul", "albania": "Tirana", "algeria": "Algiers", "andorra": "Andorra la Vella",
    "angola": "Luanda", "argentina": "Buenos Aires", "armenia": "Yerevan", "australia": "Canberra",
    "austria": "Vienna", "azerbaijan": "Baku", "bahamas": "Nassau", "bahrain": "Manama",
    "bangladesh": "Dhaka", "barbados": "Bridgetown", "belarus": "Minsk", "belgium": "Brussels",
    "belize": "Belmopan", "bhutan": "Thimphu", "bolivia": "Sucre & La Paz",
    "bosnia and herzegovina": "Sarajevo", "botswana": "Gaborone", "brazil": "Brasília",
    "brunei": "Bandar Seri Begawan", "bulgaria": "Sofia", "cambodia": "Phnom Penh", "cameroon": "Yaoundé",
    "canada": "Ottawa", "chile": "Santiago", "china": "Beijing", "colombia": "Bogotá",
    "costa rica": "San José", "croatia": "Zagreb", "cuba": "Havana", "cyprus": "Nicosia",
    "czech republic": "Prague", "czechia": "Prague", "denmark": "Copenhagen",
    "dominican republic": "Santo Domingo", "ecuador": "Quito", "egypt": "Cairo",
    "el salvador": "San Salvador", "estonia": "Tallinn", "ethiopia": "Addis Ababa", "fiji": "Suva",
    "finland": "Helsinki", "france": "Paris", "georgia": "Tbilisi", "germany": "Berlin",
    "ghana": "Accra", "greece": "Athens", "guatemala": "Guatemala City", "haiti": "Port-au-Prince",
    "honduras": "Tegucigalpa", "hungary": "Budapest", "iceland": "Reykjavik", "india": "New Delhi",
    "indonesia": "Jakarta", "iran": "Tehran", "iraq": "Baghdad", "ireland": "Dublin",
    "israel": "Jerusalem", "italy": "Rome", "jamaica": "Kingston", "japan": "Tokyo",
    "jordan": "Amman", "kazakhstan": "Astana", "kenya": "Nairobi", "kuwait": "Kuwait City",
    "kyrgyzstan": "Bishkek", "laos": "Vientiane", "latvia": "Riga", "lebanon": "Beirut",
    "libya": "Tripoli", "liechtenstein": "Vaduz", "lithuania": "Vilnius",
    "luxembourg": "Luxembourg City", "madagascar": "Antananarivo", "malaysia": "Kuala Lumpur",
    "maldives": "Malé", "malta": "Valletta", "mauritius": "Port Louis", "mexico": "Mexico City",
    "moldova": "Chișinău", "monaco": "Monaco", "mongolia": "Ulaanbaatar", "montenegro": "Podgorica",
    "morocco": "Rabat", "mozambique": "Maputo", "myanmar": "Naypyidaw", "namibia": "Windhoek",
    "nepal": "Kathmandu", "netherlands": "Amsterdam", "new zealand": "Wellington",
    "nicaragua": "Managua", "nigeria": "Abuja", "north korea": "Pyongyang",
    "north macedonia": "Skopje", "norway": "Oslo", "oman": "Muscat", "pakistan": "Islamabad",
    "panama": "Panama City", "paraguay": "Asunción", "peru": "Lima", "philippines": "Manila",
    "poland": "Warsaw", "portugal": "Lisbon", "qatar": "Doha", "romania": "Bucharest",
    "russia": "Moscow", "rwanda": "Kigali", "saudi arabia": "Riyadh", "senegal": "Dakar",
    "serbia": "Belgrade", "singapore": "Singapore", "slovakia": "Bratislava", "slovenia": "Ljubljana",
    "somalia": "Mogadishu", "south africa": "Pretoria, Cape Town & Bloemfontein",
    "south korea": "Seoul", "spain": "Madrid", "sri lanka": "Sri Jayawardenepura Kotte",
    "sudan": "Khartoum", "sweden": "Stockholm", "switzerland": "Bern", "syria": "Damascus",
    "taiwan": "Taipei", "tajikistan": "Dushanbe", "tanzania": "Dodoma", "thailand": "Bangkok",
    "tunisia": "Tunis", "turkey": "Ankara", "turkmenistan": "Ashgabat", "uae": "Abu Dhabi",
    "united arab emirates": "Abu Dhabi", "uganda": "Kampala", "uk": "London",
    "united kingdom": "London", "ukraine": "Kyiv", "usa": "Washington, D.C.",
    "united states": "Washington, D.C.", "uruguay": "Montevideo", "uzbekistan": "Tashkent",
    "vatican city": "Vatican City", "venezuela": "Caracas", "vietnam": "Hanoi",
    "yemen": "Sana'a", "zambia": "Lusaka", "zimbabwe": "Harare"
}
SCIENCE_DB = {
    "newton first law": "Physics: An object stays at rest or in uniform motion unless acted upon by an external force.",
    "newton second law": "Physics: Force equals mass times acceleration (F = m × a).",
    "newton third law": "Physics: For every action, there is an equal and opposite reaction.",
    "newton": "Physics: Isaac Newton formulated the 3 Laws of Motion and Universal Gravitation.",
    "gravity": "Physics: Universal force of attraction between masses. On Earth, g = 9.8 m/s².",
    "force": "Physics: A push or pull measured in Newtons (N). Formula: F = m × a.",
    "speed of light": "Physics: Light travels in vacuum at 299,792,458 m/s (approx. 3 × 10⁸ m/s).",
    "light": "Physics: Electromagnetic radiation visible to the human eye (~3 × 10⁸ m/s in vacuum).",
    "speed of sound": "Physics: Sound travels in dry air at 20°C at approximately 343 m/s.",
    "kinetic energy": "Physics: Energy of a moving object. Formula: KE = ½mv².",
    "potential energy": "Physics: Stored energy due to height or position. Formula: PE = mgh.",
    "energy": "Physics: Capacity to do work; cannot be created or destroyed.",
    "relativity": "Physics: Einstein's theory linking space, time, and mass-energy (E = mc²).",
    "ohm law": "Physics: Voltage equals current times resistance (V = I × R).",
    "electricity": "Physics: Flow of electric charge (electrons) measured in Amperes.",
    "magnetism": "Physics: Force exerted by magnets and moving charges with North and South poles.",
    "momentum": "Physics: Mass in motion, calculated as mass times velocity (p = m × v).",
    "work": "Physics: Force moving an object over distance (W = F × d), measured in Joules.",
    "power": "Physics: Rate of doing work or transferring energy (P = W / t), measured in Watts.",
    "density": "Physics: Mass per unit volume (ρ = m / V).",
    "pressure": "Physics: Force per unit area (P = F / A), measured in Pascals.",
    "archimedes": "Physics: Buoyant force equals the weight of fluid displaced.",
    "pascal law": "Physics: Pressure in an enclosed fluid transmits equally in all directions.",
    "boyle law": "Physics: At constant temperature, gas pressure is inversely proportional to volume (P₁V₁ = P₂V₂).",
    "thermodynamics": "Physics: 1st Law: Energy is conserved. 2nd Law: Entropy always increases.",
    "quantum": "Physics: Mechanics describing nature at atomic and subatomic scales.",
    "black hole": "Astrophysics: Spacetime region where gravity is so strong nothing can escape.",
    "solar system": "Astronomy: Sun and 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
    "atom": "Chemistry: Basic unit of matter with a nucleus (protons & neutrons) and electrons.",
    "electron": "Chemistry & Physics: Negatively charged subatomic particle orbiting the nucleus.",
    "proton": "Chemistry & Physics: Positively charged subatomic particle inside the nucleus.",
    "neutron": "Chemistry & Physics: Neutral subatomic particle with no charge inside the nucleus.",
    "periodic table": "Chemistry: Arrangement of all 118 chemical elements by atomic number.",
    "acid": "Chemistry: Donates hydrogen ions (H⁺) in water; has a pH less than 7.",
    "base": "Chemistry: Accepts H⁺ ions or releases OH⁻ ions; has a pH greater than 7.",
    "water": "Chemistry: H₂O molecule made of two hydrogen atoms and one oxygen atom.",
    "photosynthesis": "Biology: Plants use sunlight, water, and CO₂ to make glucose and oxygen.",
    "cell": "Biology: Basic structural and functional unit of all living organisms.",
    "dna": "Biology: Deoxyribonucleic acid carrying genetic instructions for life.",
    "rna": "Biology: Ribonucleic acid helping decode DNA to build proteins.",
    "mitochondria": "Biology: Powerhouse of the cell that generates ATP energy.",
    "evolution": "Biology: Change in inherited traits of populations over generations.",
    "heart": "Biology: Four-chambered muscular organ pumping blood through the body.",
    "brain": "Biology: Central organ of the nervous system controlling thought and movement."
}


def safe_math_eval(expr):
    cleaned = expr.lower().replace("x", "*").replace("÷", "/")
    cleaned = re.sub(r'[^0-9+\-*/().%^ ]', '', cleaned).replace('^', '**')
    if not any(ch.isdigit() for ch in cleaned):
        return None
    if not any(op in cleaned for op in ['+', '-', '*', '/', '%', '**']):
        return None
    try:
        result = eval(cleaned, {"__builtins__": None}, {})
        if isinstance(result, float):
            result = int(result) if result.is_integer() else round(result, 6)
        return f"🧮 Calculation Result: {result}"
    except Exception:
        return None


def get_falcon_response(user_input):
    raw = user_input.strip()
    low = raw.lower().strip("?!., ")

    if low in ["hi", "hello", "hey", "hi falcon", "hello falcon"]:
        return "Hello! I am Falcon AI. Ask me about any country's capital, physics, science, or math!"
    if low in ["how are you", "how are you doing", "how r u"]:
        return "I am running smoothly at 100% offline power! How can I help you today?"

    math_res = safe_math_eval(raw)
    if math_res is not None:
        return math_res

    for country in sorted(CAPITALS.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(country)}\b", low):
            return f"🌍 The capital of {country.title()} is {CAPITALS[country]}."

    for topic in sorted(SCIENCE_DB.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(topic)}\b", low):
            return f"🔬 {SCIENCE_DB[topic]}"

    vec = text_to_vector(low)
    prediction = ANN.forward(vec)
    intent_idx = prediction.index(max(prediction))

    if intent_idx == 0:
        return "Hello! I am Falcon AI, your offline assistant."
    elif intent_idx == 1:
        return "I am doing fantastic! Ready for your questions."
    elif intent_idx == 2:
        return "🧮 Please enter a valid math equation (e.g., 125 * 8 + 50)."
    elif intent_idx == 3:
        return "🌍 I couldn't recognize that country. Try: 'Capital of Japan' or 'Capital of Brazil'."
    elif intent_idx == 4:
        return "🔬 Ask about gravity, Newton's laws, relativity, light, DNA, or atoms."

    return "🦅 I can help with 125+ world capitals, physics & science concepts, or math!"
async def main(page: ft.Page):
    page.title = "Falcon AI"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#090d16"
    page.padding = 0

    def show_main_screen():
        page.controls.clear()
        page.bgcolor = "#090d16"
        page.padding = 0

        header = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text("🦅", size=28),
                            ft.Text("Welcome to Falcon!", size=24, weight=ft.FontWeight.BOLD, color="#f8fafc"),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    ft.Text(
                        "Offline Neural AI • 125+ Capitals • Physics • Math",
                        size=12,
                        color="#94a3b8",
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
            ),
            padding=ft.padding.only(top=36, bottom=16, left=16, right=16),
            gradient=ft.LinearGradient(
                begin=ft.alignment.top_left,
                end=ft.alignment.bottom_right,
                colors=["#1e1b4b", "#0f172a"],
            ),
            border=ft.border.only(bottom=ft.BorderSide(1, "#334155")),
            border_radius=ft.border_radius.only(bottom_left=24, bottom_right=24),
        )

        chat_list = ft.ListView(
            expand=True,
            spacing=12,
            padding=ft.padding.symmetric(horizontal=16, vertical=12),
            auto_scroll=True,
        )

        chat_list.controls.append(
            ft.Container(
                content=ft.Text(
                    "🦅 Welcome! Ask me any country's capital, a physics law, or any math problem.",
                    color="#e2e8f0",
                    size=14,
                ),
                bgcolor="#1e293b",
                padding=14,
                border_radius=16,
                border=ft.border.all(1, "#334155"),
            )
        )

        input_box = ft.TextField(
            hint_text="Ask Falcon anything...",
            hint_style=ft.TextStyle(color="#64748b"),
            expand=True,
            border_radius=24,
            color="#f8fafc",
            bgcolor="#1e293b",
            border_color="#475569",
            focused_border_color="#f472b6",
            content_padding=ft.padding.symmetric(horizontal=18, vertical=12),
        )

        def send_message(text_to_send=None):
            msg = (text_to_send if text_to_send is not None else input_box.value).strip()
            if not msg:
                return

            chat_list.controls.append(
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Text(msg, color="#ffffff", size=14),
                            gradient=ft.LinearGradient(colors=["#db2777", "#9333ea"]),
                            padding=12,
                            border_radius=ft.border_radius.only(
                                top_left=16, top_right=16, bottom_left=16, bottom_right=4
                            ),
                        )
                    ],
                    alignment=ft.MainAxisAlignment.END,
                )
            )

            if text_to_send is None:
                input_box.value = ""

            reply = get_falcon_response(msg)

            chat_list.controls.append(
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Text(reply, color="#f1f5f9", size=14),
                            bgcolor="#1e293b",
                            border=ft.border.all(1, "#334155"),
                            padding=12,
                            border_radius=ft.border_radius.only(
                                top_left=16, top_right=16, bottom_right=16, bottom_left=4
                            ),
                            expand=True,
                        )
                    ],
                    alignment=ft.MainAxisAlignment.START,
                )
            )
            page.update()

        input_box.on_submit = lambda e: send_message()

        send_btn = ft.Container(
            content=ft.Text("Send", color="#ffffff", weight=ft.FontWeight.BOLD),
            gradient=ft.LinearGradient(colors=["#ec4899", "#8b5cf6"]),
            padding=ft.padding.symmetric(horizontal=18, vertical=13),
            border_radius=24,
            on_click=lambda e: send_message(),
        )

        quick_chips = ft.Row(
            [
                ft.Container(
                    content=ft.Text("🌍 Capital of India", size=12, color="#cbd5e1"),
                    bgcolor="#1e293b",
                    padding=ft.padding.symmetric(horizontal=10, vertical=6),
                    border_radius=14,
                    border=ft.border.all(1, "#334155"),
                    on_click=lambda e: send_message("Capital of India"),
                ),
                ft.Container(
                    content=ft.Text("⚡ Relativity", size=12, color="#cbd5e1"),
                    bgcolor="#1e293b",
                    padding=ft.padding.symmetric(horizontal=10, vertical=6),
                    border_radius=14,
                    border=ft.border.all(1, "#334155"),
                    on_click=lambda e: send_message("Relativity"),
                ),
                ft.Container(
                    content=ft.Text("🧮 45 * 12", size=12, color="#cbd5e1"),
                    bgcolor="#1e293b",
                    padding=ft.padding.symmetric(horizontal=10, vertical=6),
                    border_radius=14,
                    border=ft.border.all(1, "#334155"),
                    on_click=lambda e: send_message("45 * 12"),
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=8,
        )

        bottom_bar = ft.Container(
            content=ft.Column([quick_chips, ft.Row([input_box, send_btn], spacing=8)], spacing=8),
            padding=ft.padding.only(left=14, right=14, top=10, bottom=18),
            bgcolor="#0f172a",
            border=ft.border.only(top=ft.BorderSide(1, "#1e293b")),
        )

        page.add(ft.Column([header, chat_list, bottom_bar], expand=True, spacing=0))
        page.update()

    bg_audio = None
    try:
        import flet_audio as fta
        bg_audio = fta.Audio(src="song.mp3", autoplay=True)
        page.overlay.append(bg_audio)
    except Exception:
        pass

    page.bgcolor = "#000000"
    page.padding = 0

    try:
        import flet_video as ftv
        petals_vid = ftv.Video(
            playlist=[ftv.VideoMedia("petals.mp4")],
            autoplay=True,
            show_controls=False,
            expand=True,
            fit=ft.BoxFit.COVER,
        )
        page.add(petals_vid)
        page.update()
        await asyncio.sleep(6)
    except Exception:
        page.add(
            ft.Container(
                content=ft.Text("🦅 Loading Falcon AI...", color="#ffffff", size=22, weight=ft.FontWeight.BOLD),
                alignment=ft.alignment.center,
                expand=True,
            )
        )
        page.update()
        await asyncio.sleep(2)

    try:
        if bg_audio:
            bg_audio.pause()
        page.overlay.clear()
    except Exception:
        pass

    show_main_screen()


ft.run(main, assets_dir="assets")
