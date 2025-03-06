import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn import LayerNorm
from transformers import BertTokenizer, BertForQuestionAnswering


pretrained_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


bert_state_dict = pretrained_model.state_dict()


def extract_params(state_dict, key):
    return torch.nn.Parameter(state_dict[key], requires_grad=False)


def token_embedding(input_ids, token_type_ids, params):
  
    word_embeds = F.embedding(input_ids, params['bert.embeddings.word_embeddings.weight'])
    
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
    pos_embeds = F.embedding(position_ids, params['bert.embeddings.position_embeddings.weight'])
 
    segment_embeds = F.embedding(token_type_ids, params['bert.embeddings.token_type_embeddings.weight'])
  
    embeddings = word_embeds + pos_embeds + segment_embeds
   
    embeddings = layer_norm(embeddings, params['bert.embeddings.LayerNorm.weight'], params['bert.embeddings.LayerNorm.bias'])
    return embeddings


def layer_norm(x, weight, bias, eps=1e-12):
    ln = LayerNorm(x.size(-1), eps=eps, elementwise_affine=False)
    ln.weight = torch.nn.Parameter(weight)
    ln.bias = torch.nn.Parameter(bias)
    return ln(x)



def scaled_dot_product_attention(query, key, value, attn_mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))
    
    attn_probs = F.softmax(scores, dim=-1)
    return torch.matmul(attn_probs, value)


def bert_encoder(x, params, attn_mask):
    for i in range(24): 
      
        q_w, q_b = params[f'bert.encoder.layer.{i}.attention.self.query.weight'], params[f'bert.encoder.layer.{i}.attention.self.query.bias']
        k_w, k_b = params[f'bert.encoder.layer.{i}.attention.self.key.weight'], params[f'bert.encoder.layer.{i}.attention.self.key.bias']
        v_w, v_b = params[f'bert.encoder.layer.{i}.attention.self.value.weight'], params[f'bert.encoder.layer.{i}.attention.self.value.bias']
        
       
        query = F.linear(x, q_w, q_b)
        key = F.linear(x, k_w, k_b)
        value = F.linear(x, v_w, v_b)
        
       
        batch_size, seq_len, hidden_size = query.size()
        num_heads = 16  
        head_dim = hidden_size // num_heads
        
        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
      
        attn_output = scaled_dot_product_attention(query, key, value, attn_mask=attn_mask.unsqueeze(1))
        
    
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output_w, output_b = params[f'bert.encoder.layer.{i}.attention.output.dense.weight'], params[f'bert.encoder.layer.{i}.attention.output.dense.bias']
        attn_output = F.linear(attn_output, output_w, output_b)
        
       
        norm_w, norm_b = params[f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight'], params[f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias']
        x = layer_norm(x + attn_output, norm_w, norm_b)
        
      
        ff_w1, ff_b1 = params[f'bert.encoder.layer.{i}.intermediate.dense.weight'], params[f'bert.encoder.layer.{i}.intermediate.dense.bias']
        ff_w2, ff_b2 = params[f'bert.encoder.layer.{i}.output.dense.weight'], params[f'bert.encoder.layer.{i}.output.dense.bias']
        
        ff_output = F.gelu(F.linear(x, ff_w1, ff_b1))
        ff_output = F.linear(ff_output, ff_w2, ff_b2)
       
        norm_w2, norm_b2 = params[f'bert.encoder.layer.{i}.output.LayerNorm.weight'], params[f'bert.encoder.layer.{i}.output.LayerNorm.bias']
        x = layer_norm(x + ff_output, norm_w2, norm_b2)
        
        
        
    return x


def functional_bert_qa(question, context, params):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]  
    attention_mask = inputs["attention_mask"].bool()
    


    embeddings = token_embedding(input_ids, token_type_ids, params)  
    encoded_output = bert_encoder(embeddings, params, attention_mask)

   
    qa_outputs = F.linear(encoded_output, params['qa_outputs.weight'], params['qa_outputs.bias'])
    
    
    start_logits = qa_outputs[:, :, 0]  
    end_logits = qa_outputs[:, :, 1]    
    
    print("Start Logits:", start_logits)
    print("End Logits:", end_logits)
    
    start_idx = torch.argmax(start_logits, dim=1).item()  
    end_idx = torch.argmax(end_logits, dim=1).item()      
    
    if end_idx < start_idx:
        end_idx = start_idx
        
    
    print("Start index:", start_idx)
    print("End index:", end_idx)

   
    answer_tokens = input_ids[0][start_idx:end_idx+1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)  
    

    return answer if answer.strip() else "No answer found"

