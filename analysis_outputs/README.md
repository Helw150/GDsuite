# Delphi GDsuite Analysis Outputs

This directory contains generated analysis artifacts for the Delphi GDsuite
scaling plots.

Included artifacts:

- `plot_delphi_blog_metrics_plotly.py`: regenerates the Plotly figures from the
  Hugging Face results dataset.
- `delphi_blog_accuracy_like_family_means.csv`: per-family summary table using
  hard accuracy for logprob-style benchmark families and persona match rate for
  persona QA.
- `delphi_blog_prob_margin_family_means.csv`: per-family summary table using
  probability margin, defined as `P(expected) - P(parrot)`, and persona match
  rate for persona QA.
- `delphi_plotly_blog_*`: full-grid and collection-optima Plotly exports.
- `delphi_twitter_blog_*`: 1600x900 collection-optima exports intended for
  social sharing.

The HTML exports use a public Plotly CDN script. No API tokens or private URLs
are required to view the generated figures.
