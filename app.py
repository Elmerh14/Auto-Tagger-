from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = Flask(__name__)

# ------------------
# Load Model
# ------------------
model_path = "autotagger-ner-model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label = model.config.id2label


# Extract the entities 
def extract_entities(text):
    tokens = text.split()

    # Tokenize for model input
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        preds = outputs.logits.argmax(dim=-1)[0].tolist()

    # Tokenize again (CPU) just to get word_ids
    raw = tokenizer(tokens, is_split_into_words=True)
    word_ids = raw.word_ids()

    entity_map = {"MAKE": [], "MODEL": [], "YEAR": [], "PART": [], "BRAND": []}
    curr_tokens = []
    curr_type = None
    prev_word_id = None

    for pred_id, word_id in zip(preds, word_ids):
        # Skip special tokens & duplicate token pieces
        if word_id is None or word_id == prev_word_id:
            prev_word_id = word_id
            continue
        prev_word_id = word_id

        label = id2label[pred_id]
        word = tokens[word_id]

        # Begin entity
        if label.startswith("B-"):
            if curr_tokens and curr_type:
                entity_map[curr_type].append(" ".join(curr_tokens))
            curr_type = label.split("-")[1]
            curr_tokens = [word]

        # Continue entity
        elif label.startswith("I-"):
            typ = label.split("-")[1]
            if curr_type == typ:
                curr_tokens.append(word)
            else:
                # If model gave a weird I- label after wrong type, start new
                if curr_tokens:
                    entity_map[curr_type].append(" ".join(curr_tokens))
                curr_type = typ
                curr_tokens = [word]

        # Outside entity
        else:
            if curr_tokens and curr_type:
                entity_map[curr_type].append(" ".join(curr_tokens))
            curr_tokens = []
            curr_type = None

    # Close last entity
    if curr_tokens and curr_type:
        entity_map[curr_type].append(" ".join(curr_tokens))

    # Return FIRST instance per entity
    return {k: (v[0] if v else None) for k, v in entity_map.items()}



# Run the flask browser 
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        user_input = request.form.get("listing")
        result = extract_entities(user_input)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
