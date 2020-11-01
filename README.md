*Note: in case you somehow haven't heard, GPT-2 has been superceded by GPT-3, which is much better at the whole text-generation-without-fine-tuning thing.  But this script might still be handy if you want something that you can run locally, or if you don't have access to the GPT-3 API.*

# Generate Things Using GPT-2

This is a script I use for generating lists of things (e.g. headlines, video
game achievements, character names, etc.) using GPT-2. 

This uses the GPT-2 large model (774M parameters) from OpenAI, via the
library by [huggingface](https://github.com/huggingface/transformers),
to quickly generate lists of things.  There is no fine-tuning step, so you
don't need a large corpus of whatever you're trying to generate, or an absurd
amount of computing power (however, you will still want to use a GPU to achieve
bearable speeds when generating text).  This approach
is mostly limited to generating short text excerpts
(sentences or small paragraphs), and the
quality will vary greatly depending on how well represented your target
domain is in the original training data.

It's pretty hacky, and there are tons of other people on the internet sharing
scripts for GPT-2 text generation, so I haven't really bothered to
make this user-friendly or document it, but I figured I might as well make
it public.

## Instructions
You will need to have pytorch and CUDA installed.

To control what type of thing is generated, create a context file, like
the example `news.txt`:
```
Breaking news
America invades Vancouver Island, sparking world war fears
Forest fires spring up all across the Pacific ocean; Experts baffled
Studies show that eating rocks may reduce cancer risk
Leaks suggest that the next iPhone will have twelve cameras
Burkina Faso wins bid to host 2028 Olympics
```

*This is not training data*, so you don't need a ton of examples, just enough
to give the model some context about what you want it to generate.

To run the script:
`python generate_things.py news.txt`

(There are optional arguments as well.  Run `python generate_things.py --help` to list them.  I may get around to documenting them eventually.)

The model will repeatedly be shown a context along the lines of:
```
Breaking news: Studies show that eating rocks may reduce cancer risk
Breaking news: Burkina Faso wins bid to host 2028 Olympics
Breaking news: America invades Vancouver Island, sparking world war fears
Breaking news: 
```

The model will then complete the line in a reasonable way (hopefully), and
the results will be written intermittently to a text file.  The results should
resemble the provided `example_generated_fake_news.txt`.

## Copyright
Since some of the functions are partially copied (with modification) from the
[huggingface transformers](https://github.com/huggingface) repo's examples,
this is a derivative work of
that one (which was released under the Apache License, version 2.0).
For simplicity, I've decided to release this work under the same license,
even though I think that's not technically required?
See `LICENSE.txt`.
