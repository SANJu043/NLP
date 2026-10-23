# Telugu Text Preprocessing & Tokenization Pipeline

This project builds a text cleaning and tokenization pipeline for Telugu, working off the `ai4bharat/IndicCorpV2` corpus. The end goal is simple to state and a little annoying to actually do well: take raw, messy, web-scraped Telugu text and turn it into clean, sentence-split, word-tokenized data that's usable for downstream NLP work (language modeling, statistics, whatever comes next).

Telugu doesn't have the luxury of mature, battle-tested tokenizers the way English does, so a fair chunk of this notebook is really about handling the small, annoying edge cases that a naive `text.split()` or a plain regex would get wrong — things like Telugu digits, mixed Telugu-English abbreviations, decimal numbers, and Unicode characters that look identical but aren't.

---

## What's actually in this notebook

The notebook is written as a working lab notebook, not a polished script — meaning it builds things up step by step, tests them on small samples first, and only runs the expensive full-scale version once the logic has been verified. Below is the story of what happens, in order, and why.

### 1. Pulling the data

We use `ai4bharat/IndicCorpV2`, specifically the `tel_Telu` (Telugu) split, loaded through Hugging Face's `datasets` library in **streaming mode**.

**Why streaming?** The full corpus is huge — tens of millions of lines. Downloading it all to disk before doing anything with it would be slow and wasteful, especially while we're still figuring out whether our cleaning logic even works. Streaming lets us pull records one at a time as an iterable, so we can preview a handful of examples, test our functions, and only commit to writing large volumes of data once we trust the pipeline.

A small sample (first 20,000 lines, later narrowed to 10,000 for the main experiments) is written to `data/raw/telugu_sample_50k.txt`. This sample is what all the early development and testing happens against, before the same logic gets pointed at the full corpus.

### 2. Cleaning the text

Raw web text is messy — extra whitespace, inconsistent digit systems, repeated punctuation, and Unicode characters that render the same but are encoded differently. Each cleaning step exists because it fixes one specific, observed problem:

- **Removing empty records.** Some entries in the corpus are just blank strings. These get filtered out early so they don't pollute sentence counts or waste processing time downstream.

- **Unicode normalization (NFC).** Telugu script, like many Indic scripts, can represent the same visual character using different underlying byte sequences (a base character plus combining marks, vs. a single precomposed character). Without normalizing to NFC, two strings that *look* identical could be treated as different strings by any downstream comparison, deduplication, or tokenization step. This is a classic silent bug in Indic NLP if you skip it.

- **Collapsing whitespace.** Scraped text often has irregular spacing, double spaces, tabs, or line-break artifacts. A simple regex collapses any run of whitespace into a single space, which keeps tokenization from producing empty or junk tokens later.

- **Telugu digits to Arabic digits.** Telugu has its own digit glyphs (౦-౯) alongside standard Arabic numerals (0-9), and both show up in real text. Standardizing everything to Arabic digits means numbers behave consistently for any later numeric parsing, rather than having two parallel digit systems to handle everywhere.

- **Collapsing repeated punctuation.** Text like `"అది వచ్చింది!!!"` or `"ఇది...."` gets normalized so repeated punctuation marks (`!!!`, `...`) collapse to a single mark. This keeps sentence-boundary detection from getting confused by decorative or emphatic punctuation that isn't actually marking multiple sentence ends.

All of these cleaning functions are applied with `dataset.map(...)`, which fits naturally with the streaming dataset object and keeps the pipeline lazy (nothing is actually computed until we iterate over it).

### 3. Word tokenization: the regex-based tagger

This is the heart of the pipeline. Rather than splitting purely on whitespace (which breaks the moment punctuation is glued to a word) or using an off-the-shelf English tokenizer (which doesn't understand Telugu script ranges at all), we built a **single master regex** with named capture groups, each one representing a token category:

- `url`, `email` — so links and addresses aren't shredded into meaningless fragments
- `date` — patterns like `8-5-2021`
- `comma_num` — Indian-style grouped numbers like `1,00,000`
- `decimal` — numbers like `9.80`
- `num_suffix` — a number directly fused with a Telugu suffix, like `18న` (very common in Telugu, where case-like suffixes attach directly to numerals)
- `number` — plain integers
- `word` — any run of Telugu or English letters (including Unicode joiner characters `\u200C`/`\u200D`, which are needed for correctly rendering certain Telugu conjuncts)
- `punct` — anything left over, treated as a single punctuation/symbol character

**Why one big regex instead of several smaller passes?** Because token boundaries interact. If you strip punctuation first and tokenize by whitespace second, you lose the ability to tell "was this a decimal number or a sentence-ending period followed by a number?" Doing it all in one pass, ordered from most-specific pattern to least-specific, lets each character get correctly claimed by the *right* category before a more generic fallback grabs it.

The regex alternation order matters a lot here — `url`/`email`/`date`/`comma_num`/`decimal`/`num_suffix` are all checked *before* the generic `number` and `word` patterns, so a decimal number doesn't accidentally get split into two separate integer tokens around the dot.

### 4. Sentence tokenization: not just "split on period"

Splitting Telugu text into sentences by simply cutting at every `.`, `?`, or `!` fails constantly, because periods show up inside abbreviations (`డా.` for "Dr.", `కి.మీ.` for "km"), inside decimal numbers, inside dates, and inside URLs — none of which are sentence boundaries.

So `sentence_tokenize` does something a bit more careful:

1. It reuses the same master regex to find "protected spans" — substrings like URLs, emails, dates, decimals, and comma-formatted numbers — where any punctuation inside them should *never* be treated as a sentence boundary.
2. It adds a second protected list: a hand-curated set of common Telugu and English abbreviations (`కి.మీ.`, `డా.`, `Dr.`, `Mr.`, `U.S.`, etc.), because these are exactly the places a naive splitter breaks.
3. It scans for actual sentence-ending punctuation (`.`, `?`, `!`, and the ellipsis `…`), and only accepts a position as a real sentence boundary if it doesn't fall inside one of those protected spans.
4. It also "swallows" any closing quotation marks that immediately follow a sentence-ending mark, so a sentence ending in a quoted phrase doesn't get split awkwardly right after the punctuation but before the closing quote.

This was tested directly against real, gnarly examples pulled from the corpus (postal-distance sentences with `కి.మీ.`, quoted temple names, ellipsis usage) rather than synthetic toy examples, specifically because those are the patterns that broke naive approaches during development.

### 5. Putting it together: paragraph to tokens

`tokenize_paragraph` and `paragraph_to_output_lines` chain the two pieces above: first split a paragraph into sentences, then tokenize each sentence into words, and finally join the tokens back together with single spaces. This space-joined format is deliberate — it's a simple, inspectable, whitespace-tokenized text format that's easy to sanity-check by eye and easy to feed into almost any downstream tool without extra parsing.

### 6. Running it at scale, and measuring the result

Once the logic was validated on the 10,000-line sample, the exact same cleaning + tokenization functions were consolidated into a single `clean_text()` function and re-run against the **full corpus** (over 21 million lines, ~83 million sentences, ~737 million word tokens) inside Google Colab, writing output to Google Drive. This step alone took a little over two hours to run, which is exactly why the smaller sample was used for all the iterative development — nobody wants to debug a regex against a two-hour job.

Output is saved in two formats for two different reasons:
- **Parquet** (`telugu_tokenized_full.parquet`) — compact, columnar, and fast to load partially (you can read just the first few rows of a multi-gigabyte file without loading the whole thing into memory, which is exactly what the later inspection cells do with `pyarrow.ParquetFile.iter_batches`).
- **Plain `.txt`** — because sometimes you just want to `cat` or `less` the file and eyeball it, and not every tool or grader wants to deal with Parquet.

Basic corpus statistics are computed and saved alongside the data:
- Total sentences, total words, total characters
- Average sentence length (words per sentence)
- Average word length (characters per word)
- Type/Token Ratio (TTR) — the ratio of unique tokens to total tokens, which is a rough proxy for vocabulary diversity relative to corpus size

One thing worth being upfront about: the TTR reported at the small-sample scale (10K lines, ~0.19) and the TTR at full-corpus scale (~0.013) are not directly comparable, and that's expected, not a bug. TTR naturally shrinks as corpus size grows, because you keep encountering the same common words over and over while new unique words appear more slowly. It's a well-known quirk of the metric, not a sign the pipeline broke.

---

## Requirements

```
datasets
tqdm
pandas
pyarrow
```

Install with:
```bash
pip install datasets tqdm pandas pyarrow
```

## Design choices worth remembering

A quick summary of the "why," for anyone revisiting this later:

| Choice | Reason |
|---|---|
| Streaming dataset load | Avoid downloading tens of millions of lines before the pipeline is even validated |
| Small sample first, full corpus later | The full run takes ~2+ hours; iterate cheaply, commit expensively |
| NFC Unicode normalization | Prevents visually-identical Telugu characters from being treated as different strings |
| Telugu → Arabic digit mapping | One consistent number system instead of two parallel ones |
| Single master regex with named groups | Token category decisions need to happen together, not in separate passes, because they interact (e.g. decimals vs. sentence-ending periods) |
| Abbreviation + protected-span aware sentence splitting | Naive "split on period" breaks constantly on real Telugu text (abbreviations, dates, decimals, URLs) |
| Dual output format (Parquet + txt) | Parquet for efficient large-scale storage/partial reads, txt for easy manual inspection |
| Reported TTR at two different scales | TTR is corpus-size dependent; the two numbers aren't meant to match, and that's expected behavior, not an error |

## Known limitations / things to keep in mind

- The abbreviation list is hand-curated and not exhaustive — any abbreviation not on that list will still be treated as a sentence boundary if it ends in a period.
- The tokenizer is regex-based, not statistical or learned, so it will not handle every possible edge case in informal or highly irregular text (typos, code-switching mid-word, unusual symbol usage).
- The "punct" category treats every non-word, non-number character as a separate single-character token, which is a simple and predictable rule but doesn't group multi-character symbols (e.g. emoji sequences) intelligently.
