# Decision models (`decide` and `decision`)

Decision models answer **set questions about a text**, such as "which mood?",
"how urgent, on a scale?" or "is a refund requested?". They return a
probability for every allowed answer and write no text, so an answer takes a
fraction of a second. Everything runs on this Mac.

Several families exist, from small and fast for short texts to large ones that
read long texts and decide many questions together. `astronomical models
list` and `astronomical models supported` say which this Mac has and which can
be downloaded; start with the smallest one that answers well.

## Asking questions (`astronomical decide`)

The text being judged comes from the command line, a file or standard input.
Ask one yes-or-no question directly, or pass typed questions as a JSON object
of question name to question. The help lists the exact flags.

```json
{
  "mood":    {"type": "choice", "instructions": "What is the main feeling of this line?",
              "criteria": {"love": "romantic love or longing", "grief": "sorrow or mourning",
                           "jest": "joking or teasing"}},
  "urgency": {"type": "score",  "instructions": "How urgent is this?",
              "criteria": ["not urgent", "somewhat", "very urgent"]},
  "refund":  {"type": "noul",   "instructions": "Is a refund requested?"}
}
```

- **`choice`** picks one option; a short description per option helps.
- **`score`** picks one level of an ordered scale, lowest first.
- **`noul`** is yes or no.

The answer is one JSON document with, for each question, the chosen answer
and a probability for every option. Read the rest from a real answer.

- If the answer reports that the model could not tell all options apart, or
  that the text was cut, shorten the options or the text.
- Thresholds such as "act only above 0.9" are the caller's choice.

Programs use the same request over the local address; see
`references/local-api.md`.

## Making a model fit the person's data (`astronomical decision`)

There are three levels. Each writes a **new model folder** and never changes
the original. Any amount of data is accepted; the folder is written whatever
the result, with a verdict that says what the evidence showed. Go up a level
only when the one below is not enough.

| Level | What changes | Use when |
|---|---|---|
| Calibrate | only how sure the answers sound, never which answer is picked | answers are right but too sure or too timid |
| Train the final layers | how the model weighs what it reads | it notices what matters but weighs it wrongly |
| Train the whole model | every layer, through small add-ons | it does not notice what matters at all |

The help's `decision` entries say which models each level supports and how
long training takes.

### The examples file

One JSON object per line: the `decide` request plus the right answer for
every question under `labels`.

```json
{"state": "O happy dagger! This is thy sheath.", "questions": {"mood": {"type": "choice", "instructions": "What is the main feeling?", "criteria": ["love", "grief", "jest"]}, "refund": {"type": "noul", "instructions": "Is a refund requested?"}}, "labels": {"mood": "grief", "refund": false}}
```

- A `choice` label is the option's name, a `score` label is the level's
  position (0 is the lowest), and a `noul` label is `true` or `false`.
- Every question must be labelled; a typo is refused, not ignored.
- Examples about **many different texts** matter more than many questions
  about one text: only different texts can be held back to judge the result.

### Reading the result

Training prints progress, then a report ending in a verdict:

- **Judged better:** it improved on examples held back from training.
- **Judged not better:** the report lists what got worse or did not improve.
  Not a bug; try more varied examples or a lower level.
- **Not judged:** too few different texts to hold any back, so it was only
  compared on what it learned from.

The new model is named after its output folder; pass that name to the model
option. Its answers carry the verdict with them. Deleting the folder removes
it; the original is untouched. The help's output option says where the folder
may go.

## Rules for agents

- Run `astronomical status` first. Training uses the graphics chip and swaps
  the loaded model; if the person may be using the Mac for heavy work, ask
  before training, and run one training at a time.
- Tell the person the verdict and its warnings in plain words; using the
  trained model is their decision.
- Never write into a source model folder or hand-edit what training writes.
- Train decision models only through `astronomical decision`. General
  fine-tuning tools learn next-word text and do not fit these models.
