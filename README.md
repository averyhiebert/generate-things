# Generate Things using GPT-2

The script I use for generating lists of things using GPT-2. 
What things? Any things.

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
the example `achievements.txt`:

```
Achievement
Kill a monster with your bare hands.
Find all twelve runic inscriptions.
Discover who killed your father.
Explore every room in the cursed castle.
Defeat the incarnation of Death himself.
Survive two critical attacks in a row.
Learn how to fly.
Meet the old librarian and learn his secret.
Obtain the rank of grand master blacksmith.
```

*This is not training data*, so you don't need a ton of examples, just enough
to give the model some context about what you want it to generate.

To run the script:
`python generate-things.py achievements.txt`

(There are optional arguments as well.  Run `python generate-things.py --help` to list them.  I may get around to documenting them eventually.)

The model will repeatedly be shown a context along the lines of:

```
Achievement: Learn how to fly.
Achievement: Survive two critical attacks in a row.
Achievement: Find all twelve runic inscriptions.
Achievement:
```

The model will then complete the line in a reasonable way (hopefully), and
the results will be written intermittently to a text file.
