# Writing style for the notebooks

The notebooks are a course in their own right: a reader who never saw the diploma's material must be able to learn from them. Canonical example: `src/authoring/m01/01_tables/build.py`.

## Audience

- **Assume:** basic-to-intermediate programming (Python syntax, functions, loops, data structures), calculus, linear algebra.
- **Don't assume:** probability and statistics beyond the very basics, or any pandas/NumPy/matplotlib detail beyond a first contact.
- Explain each statistics concept at the depth its difficulty calls for. Don't dwell on an average unless there's a real insight to give (e.g. "the mean minimizes squared distance"). Do define and motivate a variance with $n-1$, a quantile rule, a correlation coefficient, or a missingness mechanism.

## Self-contained

- **Define every term and acronym the first time it appears in a notebook:** full name first, then the abbreviation. For example, "the *interquartile range* (IQR)" or "*median absolute deviation* (MAD)". Do this even if an earlier notebook defined it, but keep the reminder to a clause.
- Every formula defines all its symbols, and comes with one or two sentences of **intuition**: why it has this form, and what it rewards or punishes. Use a short derivation when it's the clearest explanation (e.g. setting a derivative to zero).
- Describe the data you use: what a row is, units (BRL, days), codes (state abbreviations).
- Name the source material in the header's "Based on" row (diploma, module, course notebook names, and a link to the source repository's root, never to a branch or a folder inside it). Never copy the course's code. Use different variable names and different questions; the ideas are the same.

## Section anatomy

Each concept section has, in this order, using only the parts that help:

1. **Explanation:** the *why* first, then definitions, formulas and intuition. Keep it short. Bullets and small tables beat long paragraphs.
2. `> 🧰 **How the code works**`: the API details that matter. Defaults that surprise people (`ddof`, `dropna`, index alignment) and what a call returns.
3. `> ✅ **Good practice**` and `> ⚠️` pitfalls: one line each.
4. **Toy data**, displayed, small enough to check by hand.
5. 🔮 **Predictions** before the reveal, followed by a "verify" demo cell.
6. ✍️ **Core exercises**: toy data first, then the real data.
7. ★ **Challenges** that go beyond the course: edge cases, performance, audits, confounding, less common APIs.
8. 💬 **Explain** prompts where understanding matters more than code. The prompt is a content cell, and the answer is a separate learner cell (`nb.answer(key)`).

Each notebook also has at least one 🐞 debug exercise (code that runs but is wrong), a **mini-project** combining the notebook's concepts on one real question (with a 💬 interpretation and, ideally, an open ★), and a **wrap-up** with 4 or 5 💬 questions, `progress()`, and a pointer to the next notebook.

## Scope and size

- Organize by **topic**, not by course upload. When new course material arrives, extend the notebook whose topic it belongs to, or add a notebook.
- Each notebook: about 12–16 exercises plus a few predictions, roughly 60 % core and 40 % ★, at most about 2 hours.
- Once a skill has been taught, later notebooks **use it as a tool** without re-teaching it or re-drilling it. A one-line pointer such as "(Notebook 01)" is enough. It can come back as one step inside a bigger exercise.

## Accuracy

- **Every claim about the real data is computed before it's written**: counts, shares, "most X are Y". Print the numbers, then write the sentence.
- Every check's failure message diagnoses the likely mistake. Every success message has a one-line 💡 takeaway that generalizes the lesson.
- English throughout. Plain, direct sentences.
