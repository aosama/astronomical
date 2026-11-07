/**
 * One image generation request. The daemon's image endpoint is strict: it
 * rejects any field it does not know, takes sides between 64 and 1024 pixels
 * in multiples of 16, and answers only in base64.
 */
export interface ImageGenerationRequestBody {
  model: string;
  prompt: string;
  width: number;
  height: number;
  response_format: "b64_json";
  n: 1;
}
