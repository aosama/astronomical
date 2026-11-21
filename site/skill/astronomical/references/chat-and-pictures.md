# Asking a model, and describing pictures

`astronomical respond` sends one prompt to a chat model and prints the answer.
Its options (choosing a model, attaching pictures, instructions, a cap on
thinking, a JSON-only answer) are in `astronomical --help`.

## Describing a picture

Use this when you cannot see an image yourself, or the person wants the
description made on their Mac.

1. Check the file exists. Use a chat model that accepts pictures: the
   address's `/v1/models` list says which do, and a model that cannot is
   refused with a message saying so.
2. Run one command per picture, attaching it with the image option from the
   help, and ask a specific question when the person has one: "Read the text
   in this screenshot", "What error does this dialog show?".
3. Report the answer as the model's description, not as something you saw.

## Getting a structured answer

When a program needs the answer in a fixed shape, build a JSON schema with
`astronomical schema object` and pass it to `respond`; the reply is one JSON
object matching it.

## Good to know

- The first answer after a model swap waits for the model to load.
- A thinking model may reason for a while before the first word.
