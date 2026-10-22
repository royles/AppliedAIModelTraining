/** Mirrors backend recommended_min_corpus_chars — no network. */
export function recommendedMinCorpusChars(vocabSize: number): number {
  return Math.max(10_000, vocabSize * 25);
}
