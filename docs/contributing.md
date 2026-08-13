# Contributing

## Requirements

Follow the [instructions](https://huggingface.co/docs/reachy_mini/SDK/installation) to install the SDK. Please note that the Linux users have to [manually install gstreamer](https://huggingface.co/docs/reachy_mini/SDK/gstreamer-installation).

## Code quality

The quality of code is insured by ruff and mypy. Please make sure to run them before pushing your code.

All the tools are available in the [dev] dependency group, so you can install them with:

```bash
uv sync --group dev
```

### Ruff

A good practice is to install the pre-commit hook to run ruff on the staged files before pushing. You can do it by running the following command at the root of the repository:

```bash
pre-commit install
```

or you can run ruff manually with:

```bash
ruff check .
```

### Mypy

To check the type annotations of the code, you can run mypy with the following command:

```bash
mypy
```

Please note that mpy results depend on the installed package. The [CI](../.github/workflows/lint.yml) runs mypy with the full installation with uv on linux. Any differences between the CI and the local environment are probably due to missing dependencies. To ensure you have the same environment as the CI, you can install the package with all the extras with uv:

```bash
uv sync --all-extras --group dev
```

## Code testing

The code is tested with pytest. You can run the tests with the following command:

```bash
pytest
```

The [CI](../.github/workflows/pytest.yml) runs a limited number of tests due to the absence of robots. Check the available options in the pyproject.toml file to run the tests that are relevant for you.
For instance

```bash
pytest -m "audio"
```

## Issues and Pull Requests

Write concisely and get to the point. Issues, PR descriptions and review comments are read by humans, so no walls of text, no feature tours, no restating the diff in prose. If a sentence doesn't help the reader, cut it.

Search the open issues and pull requests first. Someone may already have reported the bug, asked for the feature, or started the work. Comment there rather than opening a duplicate.

If nothing matches, open an issue before writing the code. Describing the problem there first gets you feedback from the community and the maintainers on whether the change makes sense and how to approach it. That is much cheaper than finding out on a finished PR.

One PR = one issue. A problem spanning several concerns gets split into unitary issues, one PR each. Mention the issue in the description (`Closes #123`) so the two are linked.

The description must stand on its own: what's the problem, how to reproduce it, what's the solution, and how you know it works. A reviewer should not have to open the linked issue to follow the change. Drop the reproduction when there's nothing to reproduce, as is usually the case for a new feature.

Performance and robustness tests on physical robots are highly encouraged whenever they make sense. Paste the results (logs, charts, plots) and how to reproduce them as an annex below the main message. [#1294](https://github.com/pollen-robotics/reachy_mini/pull/1294), [#1343](https://github.com/pollen-robotics/reachy_mini/pull/1343) and [#1267](https://github.com/pollen-robotics/reachy_mini/pull/1267) are good examples of the style.

Title the PR like a commit: `type(scope): what it does`, e.g. `fix(media): tear down the playbin on EOS`.

Label every issue and every PR. That is how we triage and filter, so it isn't optional. Pick the kind of change (`bug`, `enhancement`, `documentation`, `ci`, `qol`) and the area it touches (`wireless`, `lite`, `simulation`, `audio`, `video`, `motors`). The issue form adds `needs-triage` on its own; a maintainer removes it after a first read, so leave it alone.

Each commit is one logical change, with a message saying what it does and why. Commits are read individually, so each must stand on its own; squash fixups and work-in-progress before asking for review. On a branch nobody has reviewed yet, rewrite history freely and `git push --force-with-lease`. Once review has started, stop rewriting, or reviewers lose their place.

If you changed a daemon route, regenerate the REST API reference (see [Generating the documentation](generate.md#regenerating-the-rest-api-reference)) and commit `docs/source/API/openapi.json`. CI fails on drift.

CI doesn't run on your push. A maintainer triggers it once the PR looks legitimate, so don't expect checks to start on their own. Once it runs, a failing CI pauses the review until you fix it.

## AI coding assistants

AI-assisted contributions are welcome, but the human author owns the result: read the code, understand it, and answer review comments in your own words.

We follow the Linux kernel convention from [coding-assistants.rst](https://github.com/torvalds/linux/blob/master/Documentation/process/coding-assistants.rst). Add one trailer per assisted commit:

```
Assisted-by: Claude:claude-opus-4-7
```

Append specialized analysis tools if you used any (`Assisted-by: Claude:claude-opus-4-7 coccinelle sparse`); don't list generic ones like git or ruff.

Say it outside the commits as well. In a PR, tick the matching box in the template's AI assistance section. In an issue, answer the AI assistance dropdown in the form. "No AI involvement" is a perfectly good answer; this is not a judgement on the contribution, it just tells reviewers how to read it.

An assistant must **never** add a `Signed-off-by` trailer, since only a human can certify the DCO. Keep agent and model names out of the PR title too.
