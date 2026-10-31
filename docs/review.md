# Anonymous review packaging

The review artifact contains method source, generic configuration, corrected cold
starts, synthetic tests and reproduction documentation. It excludes author and
affiliation metadata, paper/preprint links, personal repository links, host paths,
credentials, raw data, checkpoints, LLM transcripts and historical experiment logs.
The MIT notice is retained.

Run `python scripts/review_artifact.py` to scan distributable files, then
`python scripts/review_artifact.py --archive ../recharness-review.zip` to create a
ZIP rooted at `code/`. Packaging normalizes ZIP timestamps and permissions and does
not include Git objects, commit metadata or the repository owner's name. Keep the
ZIP outside the source checkout.

Upload the ZIP as supplementary code, or distribute the same contents through an
anonymous repository service. A hosting account, earlier commits, contributor lists,
issues, badges and redirects can expose identity even when file contents are clean.
A personal-account hosting URL must not be the review-facing link. Follow the
venue's current supplementary-material instructions when choosing the delivery route.

The automated scan covers common direct identifiers and credential formats; it is
not a guarantee against inference from public method names, code similarity or
external hosting metadata. Existing hosting history is separate from the source
snapshot and is deliberately excluded from the anonymous ZIP.
