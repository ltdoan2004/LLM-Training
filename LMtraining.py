from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=512,
    num_attention_heads=6,
    num_hidden_layers=4,
    type_vocab_size=1,
)
tokenizer = RobertaTokenizerFast.from_pretrained("./oscar_en", max_len=512)
model = RobertaForMaskedLM(config=config)
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.en.txt",
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

#training 
training_args = TrainingArguments(
    output_dir="./oscar_en",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    save_strategy='epoch',
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train