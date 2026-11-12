/**
 * The name a person reads for a Hugging Face model: the repository name
 * without its owner, as the Models page lists it. "owner/Model-4bit" reads
 * "Model-4bit"; an id with no owner is already the name.
 */
export function modelNameFromHuggingfaceID(huggingfaceID: string): string {
  const repositoryNameStart = huggingfaceID.lastIndexOf("/") + 1;
  const repositoryName = huggingfaceID.slice(repositoryNameStart);
  // "owner/" names nothing; the whole id is then the least misleading label.
  return repositoryName.length > 0 ? repositoryName : huggingfaceID;
}
