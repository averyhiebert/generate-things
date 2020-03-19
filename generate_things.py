'''
Script for easily generating a list of examples of "things" using
GPT-2.

The functions top_k_top_p_filtering and sample_sequence are modified from
examples found here:
https://github.com/huggingface/pytorch-transformers
(Particularly "examples/run_generation.py")
I think that makes this a derivative work?  Anyways, I've released
everything under the same licence, so as not to cause confusion. 

Copyright notice from source:

  Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
      http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

Copyright notice for this work:

   Copyright 2019 Avery Hiebert

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import logging
import random
import json
import re
import fire

from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

# Mostly directly copied from example.
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated but code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# Copied but simplified from demo script
def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

# Function for loading the model
def load_model(device="cpu"):
    ''' Load the GPT-2 model (This will take a long time the first time, since 
    you'll need to download the entire model), and create a 
    "generate text function" using the loaded tokenizer.'''
    print("Loading GPT-2 Model")
    shortcut = 'gpt2-large'
    tokenizer = GPT2Tokenizer.from_pretrained(shortcut)
    model = GPT2LMHeadModel.from_pretrained(shortcut)
    model.to(device)
    model.eval()
    print("Done loading.")
    
    def generate_text(context,**kwargs):
        context_tokens = tokenizer.encode(context)
        out = sample_sequence(model=model, context = context_tokens,**kwargs)
        out = out[0,len(context_tokens):].tolist()
        text = tokenizer.decode(out,clean_up_tokenization_spaces=True)
        return text

    return generate_text


def generate_examples(context_file,
        n=1000,
        save_file="generated_things.txt",
        top_k=300,
        device="cuda",
        length=80,
        temperature=1,
        num_context_examples=3):
    '''First line of context file is the name of the type of thing we're 
        generating (e.g. "Location").  Every subsequent line is a short-ish 
        description/sentence/paragraph of the sort of thing you are trying 
        to generate.  You can include as many as you want, and 
        there must be at least as many as num_context_examples. 
        
        See the included example file "locations.txt", which can be used
        as context to generate descriptions of locations suitable for some
        sort of fantasy game.'''
    with open(context_file,"r") as f:
        lines = f.readlines()
    thing_name = lines[0][-1:]
    examples = lines[1:]

    generate_text = load_model(device="cuda")

    # Pre-compile regexes, although the real speed bottleneck is the model,
    #  so this is probably pointless:
    # Regex to extract examples which fit "the pattern" that we're
    #  generating things from.
    outer_regex = re.compile("\n%s: (.+)\n"%thing_name)
    # TODO: Add some option/argument for adding additional regexes to
    #  use to filter out bad/invalid results.

    for i in range(n):
        random.shuffle(examples)
        context = "\n".join(["%s: %s" % (thing_name, example) 
            for example in examples[:num_context_examples]]) + "\n"
        result = generate_text(context,length=length,temperature=temperature,
            top_k=top_k,device=device)
        
        matches = outer_regex.findall(result)
        with open(save_file,"a") as f:
            f.writelines([m.split("<|endoftext|>")[0] + "\n" for m in matches])

        if i % 100 == 99:
            print("Done generating %d examples so far" % (i+1))


if __name__=="__main__":
    fire.Fire(generate_examples)
