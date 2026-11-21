# Drawing an image

`astronomical image` draws one image from a prompt, saves it as a PNG and
prints the path. Its options (where to save, the size, the model, a seed to
repeat a result) are in `astronomical --help`.

## Writing the prompt

It draws from words, not from a photo. For "my dog as an astronaut", ask what
the dog looks like, or describe their photo first with `astronomical respond`
(see `references/chat-and-pictures.md`) and use that description.

The person's words are the starting point, not the whole prompt.

- **Say what is in the picture first,** then the setting, the light and mood,
  then the style: "a red bicycle leaning on a stone wall, narrow street in the
  morning, soft light, watercolor".
- **Be concrete.** Colors, materials, time of day and point of view help more
  than words such as "beautiful".
- **Keep text out of images** unless asked; image models spell poorly.
- **Match the shape to the use:** wide for a desktop wallpaper, a banner or a
  slide; tall for a phone screen; square otherwise.

## Saving it

Save into the folder the person is working in, or the one they named, with a
short name that says what it is. Tell them the path the command printed.

## Good to know

- The first image after a chat swaps the chat model out, so it takes longer.
- The first use downloads the image model, which is large; ask first.
- Drawing happens on this Mac. Prefer it over online image services the
  person has not asked for.
