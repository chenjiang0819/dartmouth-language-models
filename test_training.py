"""Offline tiny GPT-2 training/serialization interface regression, not pretrained quality."""
import unittest,tempfile
from pathlib import Path
from gpt2_training import write_training_text,train,generate_text
class TrainingChecks(unittest.TestCase):
 def test_offline_trainer_save_reload(self):
  import torch
  from tokenizers import ByteLevelBPETokenizer
  from transformers import GPT2Config,GPT2LMHeadModel,GPT2Tokenizer
  torch.manual_seed(0)
  with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
   p=Path(directory);text=p/'train.txt';write_training_text(['sun moon sky '*30]*5,text)
   bpe=ByteLevelBPETokenizer();bpe.train(files=[str(text)],vocab_size=260,special_tokens=['<|endoftext|>']);bpe.save_model(str(p))
   tokenizer=GPT2Tokenizer(vocab_file=str(p/'vocab.json'),merges_file=str(p/'merges.txt'));tokenizer.pad_token=tokenizer.eos_token
   model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_positions=256,n_embd=16,n_layer=1,n_head=2,bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id))
   trainer=train(text,model,tokenizer,p/'output',num_train_epochs=1,save_steps=500,use_cpu=True)
   self.assertGreater(trainer.state.global_step,0)
   self.assertTrue((p/'output'/'config.json').exists());self.assertTrue((p/'output'/'tokenizer_config.json').exists())
   restored=GPT2LMHeadModel.from_pretrained(p/'output',local_files_only=True)
   self.assertEqual(restored.config.vocab_size,len(tokenizer))
   self.assertIsInstance(generate_text('sun',p/'output',max_new_tokens=2),str)
   with self.assertRaises(ValueError):generate_text('sun',p/'output',max_new_tokens=1000)
if __name__=='__main__':unittest.main()
