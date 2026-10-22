# ES 661 · Probabilistic Machine Learning (2023)

Course website for ES 661 Probabilistic Machine Learning at IIT Gandhinagar, Semester I 2023–24:
<https://nipunbatra.github.io/pml2023/>

The site is a [Quarto](https://quarto.org) website in the same design as
[ES 667 Deep Learning](https://nipunbatra.github.io/dl-2026/). On every push to `main`, GitHub Actions
(`.github/workflows/publish.yml`) renders it and publishes it to the `gh-pages` branch with `quarto publish`.

| File | Contents |
|---|---|
| `index.qmd` | Home: course facts, course record, module overview, questions, main topics, teaching assistants |
| `schedule.qmd` | Lectures: one row per lecture with slides, cheat sheets, recordings, notebooks and reading |
| `assessment.qmd`, `_dates.qmd` | Grading, project, assignments and every dated event |
| `assignments.qmd`, `assignments/` | Assignment list and the four assignments |
| `interactive-apps.qmd` | Interactive apps |
| `lectures/` | Lecture slides (PDF) |
| `notebooks/` | Lecture notebooks, rendered as pages |
| `grading.qmd`, `quizzes.qmd`, `faq.qmd`, `notebooks/index.qmd` | Redirects from old URLs |
| `styles.scss`, `course.scss` | Shared theme (`styles.scss` is identical to dl-2026) and small additions |
| `rail.html`, `course-theme.html`, `course-tools.html` | Side rail, light and dark switch, footer, copy-link buttons and date handling |

Preview locally with `quarto preview`.
