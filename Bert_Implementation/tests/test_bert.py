import torch
import pytest
import torch.nn.functional as F
import os
import sys
from transformers import BertTokenizer, BertForQuestionAnswering

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from reference.bert import (
    token_embedding,
    bert_encoder,
    layer_norm,
    scaled_dot_product_attention,
    extract_params,
)

pretrained_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
 
 
bert_state_dict = pretrained_model.state_dict()
params = {key: extract_params(bert_state_dict, key) for key in bert_state_dict.keys()}
 
 
question = "Who wrote Harry Potter?"
context = "J.K. Rowling is the author of the Harry Potter series."
inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
input_ids = inputs["input_ids"]
token_type_ids = inputs["token_type_ids"]
attention_mask = inputs["attention_mask"].bool()


def assert_tensors_equal(tensor1, tensor2, rtol=1e-03, atol=1e-05):
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), "Tensors are not equal"

def calculate_pcc(tensor1, tensor2):
    """
    Calculate the Pearson Correlation Coefficient (PCC) between two tensors.
    """
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)
    
   
    covariance = torch.mean((tensor1_flat - torch.mean(tensor1_flat)) * (tensor2_flat - torch.mean(tensor2_flat)))
    std1 = torch.std(tensor1_flat)
    std2 = torch.std(tensor2_flat)
    
  
    if std1 == 0 or std2 == 0:
        return 0.0
    
    
    pcc = covariance / (std1 * std2)
    return pcc.item()


def test_token_embedding():

    custom_embeddings = token_embedding(input_ids, token_type_ids, params)
    
 
    pretrained_embeddings = pretrained_model.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
    
  
    print("Custom Embeddings:", custom_embeddings)
    print("Pre-trained Embeddings:", pretrained_embeddings)
    
   
    assert custom_embeddings.shape == pretrained_embeddings.shape, "Shapes do not match"
    
  
    assert_tensors_equal(custom_embeddings, pretrained_embeddings)


def test_bert_encoder():
 
    custom_embeddings = token_embedding(input_ids, token_type_ids, params)

  
    pretrained_embeddings = pretrained_model.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
    
    attention_mask = inputs["attention_mask"].bool()

 
    custom_output = custom_embeddings
    pretrained_output = pretrained_embeddings

    for i in range(24):
        
        q_w, q_b = params[f'bert.encoder.layer.{i}.attention.self.query.weight'], params[f'bert.encoder.layer.{i}.attention.self.query.bias']
        k_w, k_b = params[f'bert.encoder.layer.{i}.attention.self.key.weight'], params[f'bert.encoder.layer.{i}.attention.self.key.bias']
        v_w, v_b = params[f'bert.encoder.layer.{i}.attention.self.value.weight'], params[f'bert.encoder.layer.{i}.attention.self.value.bias']
        
       
        query = F.linear(custom_output, q_w, q_b)
        key = F.linear(custom_output, k_w, k_b)
        value = F.linear(custom_output, v_w, v_b)
        
      
        batch_size, seq_len, hidden_size = query.size()
        num_heads = 16  # BERT-large uses 16 attention heads
        head_dim = hidden_size // num_heads
        
        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
       
        attn_output = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask.unsqueeze(1))
        
       
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output_w, output_b = params[f'bert.encoder.layer.{i}.attention.output.dense.weight'], params[f'bert.encoder.layer.{i}.attention.output.dense.bias']
        attn_output = F.linear(attn_output, output_w, output_b)
        
   
        norm_w, norm_b = params[f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight'], params[f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias']
        custom_output = layer_norm(custom_output + attn_output, norm_w, norm_b)
        
      
        ff_w1, ff_b1 = params[f'bert.encoder.layer.{i}.intermediate.dense.weight'], params[f'bert.encoder.layer.{i}.intermediate.dense.bias']
        ff_w2, ff_b2 = params[f'bert.encoder.layer.{i}.output.dense.weight'], params[f'bert.encoder.layer.{i}.output.dense.bias']
        
        ff_output = F.gelu(F.linear(custom_output, ff_w1, ff_b1))
        ff_output = F.linear(ff_output, ff_w2, ff_b2)
        
       
        norm_w2, norm_b2 = params[f'bert.encoder.layer.{i}.output.LayerNorm.weight'], params[f'bert.encoder.layer.{i}.output.LayerNorm.bias']
        custom_output = layer_norm(custom_output + ff_output, norm_w2, norm_b2)

        
        pretrained_output = pretrained_model.bert.encoder.layer[i](pretrained_output, attention_mask=attention_mask)[0]
        
        print(f"Layer {i}: Custom Output Shape: {custom_output.shape}, Pre-trained Output Shape: {pretrained_output.shape}")
        
        print("Custom Output:", custom_output)
        print("Pre-trained Output:", pretrained_output)

       
        assert_tensors_equal(custom_output, pretrained_output)

     
        pcc_score = calculate_pcc(custom_output, pretrained_output)
        print(f"Layer {i}: PCC Score: {pcc_score}")
        assert pcc_score > 0.99, f"PCC score is not close to 1.0 (actual: {pcc_score})"


def test_final_logits():
    
    custom_embeddings = token_embedding(input_ids, token_type_ids, params)

    
    custom_output = bert_encoder(custom_embeddings, params, attention_mask)

   
    qa_outputs = F.linear(custom_output, params['qa_outputs.weight'], params['qa_outputs.bias'])
    custom_start_logits = qa_outputs[:, :, 0]
    custom_end_logits = qa_outputs[:, :, 1]

  
    with torch.no_grad():
        pretrained_outputs = pretrained_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    pretrained_start_logits = pretrained_outputs.start_logits
    pretrained_end_logits = pretrained_outputs.end_logits


    assert_tensors_equal(custom_start_logits, pretrained_start_logits)
    assert_tensors_equal(custom_end_logits, pretrained_end_logits)

    
    pcc_start = calculate_pcc(custom_start_logits, pretrained_start_logits)
    pcc_end = calculate_pcc(custom_end_logits, pretrained_end_logits)
    print(f"Start Logits PCC Score: {pcc_start}")
    print(f"End Logits PCC Score: {pcc_end}")
    assert pcc_start > 0.99, f"Start Logits PCC score is not close to 1.0 (actual: {pcc_start})"
    assert pcc_end > 0.99, f"End Logits PCC score is not close to 1.0 (actual: {pcc_end})"