"""2026 causal-language-model workflow using upstream PyTorch/Hugging Face APIs."""
import os
# GPT-2 training here uses PyTorch; avoid unrelated Keras backend imports.
os.environ.setdefault("USE_TF", "0")
from pathlib import Path

def write_training_text(sentences,path):
    if not sentences: raise ValueError('training corpus is empty')
    if any(not isinstance(s,str) or not s.strip() for s in sentences): raise ValueError('empty/nontext sample')
    Path(path).write_text('\n'.join(s.replace('\n',' ') for s in sentences)+'\n',encoding='utf-8')

def train(train_file_path,model,tokenizer,output_dir,num_train_epochs=1,save_steps=500,*,use_cpu=False):
    import torch
    from transformers import Trainer,TrainingArguments
    if num_train_epochs<=0 or save_steps<=0: raise ValueError('positive epochs/save_steps required')
    block_size=min(128,model.config.n_positions)
    if block_size<2: raise ValueError('model context must support next-token learning')
    ids=tokenizer.encode(Path(train_file_path).read_text(encoding='utf-8'),add_special_tokens=False)
    blocks=[torch.tensor(ids[start:start+block_size],dtype=torch.long)
            for start in range(0,len(ids)-block_size+1,block_size)]
    if not blocks: raise ValueError(f'corpus needs at least one {block_size}-token block')
    # Fixed full blocks need no padding. The model shifts labels internally.
    dataset=[{'input_ids':block,'attention_mask':torch.ones_like(block),'labels':block.clone()}
             for block in blocks]
    args=TrainingArguments(output_dir=str(output_dir),
        per_device_train_batch_size=8,num_train_epochs=num_train_epochs,save_steps=save_steps,
        report_to=[],seed=0,use_cpu=use_cpu)
    trainer=Trainer(model=model,args=args,train_dataset=dataset)
    trainer.train();trainer.save_model(str(output_dir));tokenizer.save_pretrained(str(output_dir))
    return trainer

def generate_text(sequence,model_path,max_new_tokens=30,seed=0):
    import torch
    from transformers import GPT2LMHeadModel,GPT2Tokenizer
    if not isinstance(max_new_tokens,int) or max_new_tokens<=0: raise ValueError('positive token budget required')
    model=GPT2LMHeadModel.from_pretrained(model_path,local_files_only=True)
    tokenizer=GPT2Tokenizer.from_pretrained(model_path,local_files_only=True)
    model.eval();torch.manual_seed(seed)
    ids=tokenizer.encode(sequence,return_tensors='pt')
    if ids.numel()==0: ids=torch.tensor([[model.config.bos_token_id]])
    if ids.shape[1]+max_new_tokens>model.config.n_positions: raise ValueError("Input and generation budget exceed model context")
    with torch.no_grad(): outputs=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=max_new_tokens,do_sample=True,
        pad_token_id=model.config.eos_token_id,top_k=50,top_p=.95)
    return tokenizer.decode(outputs[0],skip_special_tokens=True)
