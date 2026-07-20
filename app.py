from flask import Flask, render_template, request
import numpy as np
import pickle
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


encoder_model = load_model("encoder_model.keras")
decoder_model = load_model("decoder_model.keras")

with open("english_tokenizer.pkl","rb") as file:
    eng_tokenizer = pickle.load(file)

with open("french_tokenizer.pkl","rb") as file:
    fr_tokenizer = pickle.load(file)

with open("config.pkl", "rb") as file:
    config = pickle.load(file)

max_eng_len = config["max_eng_len"]
max_fr_len = config["max_fr_len"]

reverse_fr_word_index = {
    number: word
    for word, number in fr_tokenizer.word_index.items()
}

def clean_text(text):
    text = text.lower()
    text = text.strip()
    text =re.sub(
        f"[{re.escape(string.punctuation)}]",
        "",
        text
    )
    text = re.sub(r"\s+"," ", text)

    return text

def translate_sentence(sentence):
    sentence = clean_text(sentence)

    sequence = eng_tokenizer.texts_to_sequences([sentence])

    padded_sequence = pad_sequences(
        sequence,
        maxlen= max_eng_len,
        padding="post"
    )

    states_value = encoder_model.predict(
        padded_sequence,
        verbose=0
    )

    start_token = fr_tokenizer.word_index["start"]
    end_token = fr_tokenizer.word_index["eos"]

    target_sequence= np.array([[start_token]])
    translated_words = []

    for _ in range(max_fr_len):
        output_tokens, state_h, state_c = decoder_model.predict(
            [target_sequence] + states_value,
            verbose=0
        )

        predicted_token = int(
            np.argmax(output_tokens[0,-1,:])

        )

        if predicted_token == 0:
            break
        
        if predicted_token == end_token:
            break
        
        predicted_word = reverse_fr_word_index.get(
                    predicted_token
        )

        if predicted_word:
            translated_words.append(predicted_word)

        target_sequence = np.array(
            [[predicted_token]]
        )

        states_value = [state_h, state_c]

    return " ".join(translated_words)


@app.route("/", methods=["GET", "POST"])
def home():
    english_text = ""
    translation = ""
    error = ""

    if request.method == "POST":
        english_text = request.form.get(
            "english_text",
            ""
        ).strip()

        if english_text == "":
            error = "Please enter an English sentence."
        else:
            translation = translate_sentence(
                english_text
            )

            if translation == "":
                translation = (
                    "The model could not translate this sentence."
                )

    return render_template(
        "index.html",
        english_text=english_text,
        translation=translation,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
    
