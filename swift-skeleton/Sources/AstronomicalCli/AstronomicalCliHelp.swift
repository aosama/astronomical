import Foundation

/** The carried help contract, ported verbatim from arguments.rs HELP_TEXT. */
enum AstronomicalCliHelp {

    static let text: String = """
Astronomical

Usage: astronomical launch [tool]
       astronomical launch opencode [--model MODEL_ID]
       astronomical respond PROMPT [--text TEXT] [--image PATH]... [--model MODEL_ID]
                                   [--instructions TEXT] [--thinking-budget N] [--schema FILE]
                                   [--no-stream]
       astronomical embed [TEXT | --file PATH] [--model MODEL_ID]
       astronomical models list | supported | default [MODEL_ID] | download MODEL_ID
       astronomical status
       astronomical schema object --name NAME (--string|--int|--double|--boolean) PROPERTY...
       astronomical validate config [--instance stable|development] [--json]
       astronomical --help
       astronomical --version

Launch a coding harness against the local Astronomical Library, manage
the model library through the daemon, or run ephemeral in-process
utilities against the instance configuration.

Commands:
  launch [tool]    Start a supported harness (OpenCode in this release)
  respond PROMPT   One-shot chat answer; add --image PATH to send a raster
                   image with the prompt; add --schema FILE to force one
                   JSON object reply; the daemon loads or downloads the
                   model automatically when it is not resident yet
  embed [TEXT]     One-shot embedding vector as one JSON document; the daemon
                   loads or downloads the model automatically when needed
  models list      Models installed on this Mac
  models supported Release catalog: what can be downloaded, with local state
  models default   Show the effective default model; with MODEL_ID, download it
                   first when this Mac lacks it, then persist it
  models download  Start (or resume) a download and wait, with live progress
  status           Worker state, resident model, default model, active download
  schema object    Build a strict JSON object schema for structured output
  validate config  Report effective values of an instance configuration

Options:
  --model MODEL_ID     Model to use (default: the daemon's effective default model)
  --no-stream          Print the finished respond answer once instead of streaming
  --text TEXT          Provide the respond prompt as a flag value instead of the positional PROMPT
  --instructions TEXT  System-prompt-style guidance applied to the respond reply
  --thinking-budget N  Cap the tokens a thinking model may spend reasoning (0-65535; default: think freely)
  --image PATH         Attach a raster image (png, jpg, jpeg, webp) to the respond prompt; repeatable
  --schema FILE        Force the respond reply to be one JSON object matching the JSON schema in FILE
  --file PATH          Embed the file's contents instead of TEXT or stdin
  --instance NAME      Which instance to inspect for validate config (default: development)
  --json               Render the validate config report as JSON
  -v, --verbose        Print launch timings on stderr
  -h, --help           Show this help
  --version            Show the CLI version

"""
}
