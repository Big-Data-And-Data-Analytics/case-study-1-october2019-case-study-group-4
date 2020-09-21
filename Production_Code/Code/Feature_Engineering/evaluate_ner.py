from spacy.scorer import Scorer
from spacy.gold import GoldParse

# Function to evaluate the model with training data
def model_evaluate(model, data):
    scorer = Scorer()
    # print(data)
    for input_, annotations in data:
        # for ent in annotations.get("entities"):
        doc_gold_text = model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annotations.get("entities"))
        pred_value = model(input_)
        scorer.score(pred_value, gold)
    return scorer.ents_per_type