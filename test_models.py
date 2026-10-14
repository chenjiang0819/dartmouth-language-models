import unittest,tempfile
from pathlib import Path
from collections import defaultdict
from ngrams import train,generate,get_next_word,preprocess
from gpt2_training import write_training_text
class Checks(unittest.TestCase):
 def test_lexical_cleaner(self):
  self.assertEqual(preprocess('RT @Alice: Héllo &amp; #WORLD 123! https://example.org/a?x=2'),'user héllo world link')
  self.assertEqual(preprocess("ＡＢＣ\ncan't 42 www.example.org and @bob"),"abc can't link and user")
  self.assertEqual(preprocess('!? 123'),'')
  with self.assertRaises(TypeError):preprocess(None)
 def test_probability_oracle(self):
  v,c,p=train(['a b','a c','a b'])
  self.assertEqual(c[2][v['a'],v['b']],2)
  self.assertAlmostEqual(p[2][v['a'],v['b']],2/3)
  self.assertAlmostEqual(p[3][v['<s>'],v['a'],v['c']],1/3)
  for table in p.values():
   totals=defaultdict(float)
   for g,x in table.items():totals[g[:-1]]+=x
   for total in totals.values():self.assertAlmostEqual(total,1)
  self.assertNotIn((v['</s>'],v['<s>']),c[2])
 def test_last_candidate_eos(self):
  v,c,p=train(['hello']);self.assertEqual(generate(vocabulary=v,probs=p,n=100),'hello')
  self.assertEqual(get_next_word({(v['<s>'],v['hello']):1},1,['<s>'],vocabulary=v),[v['<s>'],v['hello']])
 def test_backoff_and_seed(self):
  v,c,p=train(['a b','a c']);self.assertEqual(generate('a',vocabulary=v,probs=p,order=3,seed=4),generate('a',vocabulary=v,probs=p,order=3,seed=4))
  self.assertEqual(generate('unseen',vocabulary=v,probs=p,order=3),'unseen'.replace('unseen','<unk>'))
 def test_empty_and_immutable(self):
  rows=[''];v,c,p=train(rows);self.assertEqual(rows,['']);self.assertEqual(generate(vocabulary=v,probs=p),'');self.assertEqual(generate(vocabulary=v,probs=p,order=3),'')
  with self.assertRaises(ValueError):train([])
  with self.assertRaises(ValueError):generate(n=-1,vocabulary=v,probs=p)
 def test_text_boundaries(self):
  with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
   p=Path(d)/'train.txt';write_training_text(['a','b'],p);self.assertEqual(p.read_text(),'a\nb\n')
 def test_gpt2_offline_gradient(self):
  import torch
  from transformers import GPT2Config,GPT2LMHeadModel
  torch.manual_seed(0)
  model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_positions=16,n_embd=16,n_layer=1,n_head=2,bos_token_id=0,eos_token_id=1))
  ids=torch.tensor([[0,3,4,1]])
  loss=model(ids,labels=ids).loss;self.assertTrue(torch.isfinite(loss));loss.backward()
  self.assertTrue(torch.isfinite(model.transformer.wte.weight.grad).all())
if __name__=='__main__':unittest.main()
