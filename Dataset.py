from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer
oscar_en = load_dataset("nthngdy/oscar-small", language='en')

## tokenizer

paths = ['./oscar.en.txt']

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save tokenizer
tokenizer.save_model("oscar_en")

#Load vocabulary
tokenizer = ByteLevelBPETokenizer(
    "./oscar_en/vocab.json",
    "./oscar_en/merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)