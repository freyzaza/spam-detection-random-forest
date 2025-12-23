import re
import string
import emoji
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

STOPWORDS = set(StopWordRemoverFactory().get_stop_words())

STOPWORDS.update({
    "hou", "kaminski", "vince", "enron", "corp", "edu", "cc", "re", "fw", "subject",
    "email", "houston", "pm", "am", "com", "net", "org", "ltd", "co", "inc", "ect"
})

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NUM_RE = re.compile(r"\d+")
MULTISPACE_RE = re.compile(r"\s+")

if hasattr(emoji, "get_emoji_regexp"):
    EMOJI_RE = emoji.get_emoji_regexp()
    def strip_emoji(txt: str) -> str:
        return EMOJI_RE.sub(" ", txt)
else:
    def strip_emoji(txt: str) -> str:
        return emoji.replace_emoji(txt, replace=" ")

PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def clean_text(text: str) -> str:
    """
    Bersihkan teks:
    - lowercase
    - hapus url, email, emoji, angka, punctuation
    - stopword removal
    - hanya token alphabet dan panjang > 2
    """
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = strip_emoji(text)
    text = text.translate(PUNCT_TABLE)
    text = NUM_RE.sub(" ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2 and t not in STOPWORDS]

    cleaned = " ".join(tokens)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip()
    return cleaned
