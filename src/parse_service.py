from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings
from io import StringIO

from init_config import ConfigParser
from parse_corpus import get_config_modification
from data_handling.custom_conll_dataset import CustomCoNLLDataset


# service settings, can be changed with env var
class Settings(BaseSettings):
    model_dir: str = "basic_mbert/"
    lstm: bool = False

    class Config:
        env_file = ".env"
        env_prefix = 'STEPS_PARSER_'


# Response is a list of sentences as a CONLL string
class StepsParserResponse(BaseModel):
    sentences: List[str]


# Request:
# CONLL Line:
# List ['1', 'This', '_', '_', '_', '_', '0', '_', '_', 'start_char=0|end_char=4']
# CONLL Sentence:
# List of CONLL Lines
# CONLL Corpus:
# List of CONLL Sentences
class StepsParserRequest(BaseModel):
    sentences: List[List[List[str]]]


# parse settings
args = Settings()


# STEPS parser
# from parse_corpus.py and parse_raw.py
keep_columns = [9]  # keep start/end indices
config = ConfigParser.from_args(args, modification=get_config_modification(args, lstm=args.lstm))
model = config.init_model()
trainer = config.init_trainer(model, None, None)
parser = trainer.parser
annotation_layers = config["data_loaders"]["args"]["annotation_layers"]
for col in keep_columns:
    annotation_layers[col] = {"type": "TagSequence", "source_column": col}
column_mapping = {annotation_id: annotation_layer["source_column"] for annotation_id, annotation_layer in annotation_layers.items()}


# API
app = FastAPI()


@app.post("/parse")
def process(request: StepsParserRequest) -> StepsParserResponse:
    # prepare conll stream
    conll_stream = StringIO()
    for sent in request.sentences:
        for token in sent:
            print("\t".join(token), file=conll_stream)
        print(file=conll_stream)
    conll_stream.seek(0)

    # parse
    dataset = CustomCoNLLDataset.from_corpus_file(conll_stream, annotation_layers)
    results = []
    for sentence in dataset:
        parsed_sentence = parser.parse(sentence)
        for col in keep_columns or []:
            parsed_sentence.annotation_data[col] = sentence[col]
        results.append(parsed_sentence.to_conll(column_mapping))

    conll_stream.close()

    return StepsParserResponse(sentences=results)
