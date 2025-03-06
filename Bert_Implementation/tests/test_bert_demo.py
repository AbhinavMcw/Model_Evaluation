import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reference.bert import functional_bert_qa, params, tokenizer


pretrained_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


question = "Who wrote Harry Potter?"
context = "J.K. Rowling is the author of the Harry Potter series."


print("Running Custom BERT Model...")
custom_answer = functional_bert_qa(question, context, params)
print(f"Custom BERT Answer: {custom_answer}")


print("\nRunning Pre-trained BERT Model...")
inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
with torch.no_grad():
    pretrained_outputs = pretrained_model(**inputs)
    

start_logits = pretrained_outputs.start_logits
end_logits = pretrained_outputs.end_logits


start_idx = torch.argmax(start_logits, dim=1).item()
end_idx = torch.argmax(end_logits, dim=1).item()

if end_idx < start_idx:
    end_idx = start_idx


answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
pretrained_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print(f"Pre-trained BERT Answer: {pretrained_answer}")


print("\nComparison:")
print(f"Custom BERT Answer: {custom_answer}")
print(f"Pre-trained BERT Answer: {pretrained_answer}")