from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings
from io import StringIO
from fastapi.logger import logger

from init_config import ConfigParser
from parse_corpus import get_config_modification
from data_handling.custom_conll_dataset import CustomCoNLLDataset


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
    batch_size: int
    model_name: str
    lstm: bool


class StepsArgs:
    def __init__(self, model_name, lstm):
        self.model_dir = "/models/" + model_name
        self.lstm = lstm


# cache model
model_cache = {}


def load_model(request: StepsParserRequest):
    if request.model_name in model_cache:
        return model_cache[request.model_name]

    args = StepsArgs(request.model_name, request.lstm)

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
    column_mapping = {annotation_id: annotation_layer["source_column"] for annotation_id, annotation_layer in
                      annotation_layers.items()}

    model_cache[request.model_name] = (
        annotation_layers,
        parser,
        keep_columns,
        column_mapping
    )

    return model_cache[request.model_name]


# API
app = FastAPI()
print("PyDockerNotify: Container startup success")


@app.get("/textimager/ready")
def get_textimager():
    return {
        "ready": True
    }


@app.post("/parse")
def process(request: StepsParserRequest) -> StepsParserResponse:
    annotation_layers, parser, keep_columns, column_mapping = load_model(request)

    maximum_batch_size = request.batch_size
    logger.info("Using batchsize {} for this request!".format(maximum_batch_size))
    if maximum_batch_size > 128:
        maximum_batch_size = 128

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
    # Iterate over the batches, when specifying batch size one iteration is sentence by sentence
    for chunks in range(0,len(dataset),maximum_batch_size):
        partitioned_dataset = dataset[chunks:(chunks+maximum_batch_size)]
        multi_parser = parser.parse_multi(partitioned_dataset)
        for (parsed_sentence,sentence) in zip(multi_parser,partitioned_dataset):
            for col in keep_columns or []:
                parsed_sentence.annotation_data[col] = sentence[col]
            results.append(parsed_sentence.to_conll(column_mapping))

    conll_stream.close()

    return StepsParserResponse(sentences=results)
