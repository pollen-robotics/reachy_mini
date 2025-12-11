# Generating the documentation

To generate the documentation, you first have to build it. Several packages are necessary to build the doc,
you can install them with the following command, at the root of the code repository:

```bash
pip install -e . -r docs-requirements.txt
```

You will also need `nodejs`. Please refer to their [installation page](https://nodejs.org/en/download)

---

**NOTE**

You only need to generate the documentation to inspect it locally (if you're planning changes and want to
check how they look before committing for instance). You don't have to `git commit` the built documentation.

---

## Building the documentation

Once you have setup the `doc-builder` and additional packages, you can generate the documentation by
typing the following command:

```bash
doc-builder build reachy_mini docs/source/ --build_dir ~/tmp/test-build
```

You can adapt the `--build_dir` to set any temporary folder that you prefer. This command will create it and generate
the MDX files that will be rendered as the documentation on the main website. You can inspect them in your favorite
Markdown editor.

## Previewing the documentation

To preview the docs, run the following command:

```bash
doc-builder preview reachy_mini docs/source/
```

The docs will be viewable at [http://localhost:5173](http://localhost:5173). You can also preview the docs once you have opened a PR. You will see a bot add a comment to a link where the documentation with your changes lives.

---

**NOTE**

The `preview` command only works with existing doc files. When you add a completely new file, you need to update `_toctree.yml` & restart `preview` command (`ctrl-c` to stop it & call `doc-builder preview ...` again).

---

## Adding a new element to the navigation bar

Accepted files are Markdown (.md).

Create a file with its extension and put it in the source directory. You can then link it to the toc-tree by putting
the filename without the extension in the [`_toctree.yml`](https://github.com/huggingface/lerobot/blob/main/docs/source/_toctree.yml) file.

## Renaming section headers and moving sections

It helps to keep the old links working when renaming the section header and/or moving sections from one document to another. This is because the old links are likely to be used in Issues, Forums, and Social media and it'd make for a much more superior user experience if users reading those months later could still easily navigate to the originally intended information.

Therefore, we simply keep a little map of moved sections at the end of the document where the original section was. The key is to preserve the original anchor.

So if you renamed a section from: "Section A" to "Section B", then you can add at the end of the file:

```
Sections that were moved:

[ <a href="#section-b">Section A</a><a id="section-a"></a> ]
```

and of course, if you moved it to another file, then:

```
Sections that were moved:

[ <a href="../new-file#section-b">Section A</a><a id="section-a"></a> ]
```

Use the relative style to link to the new file so that the versioned docs continue to work.

For an example of a rich moved sections set please see the very end of [the transformers Trainer doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/trainer.md).

### Adding a new tutorial

Adding a new tutorial or section is done in two steps:

- Add a new file under `./source`. This file can either be ReStructuredText (.rst) or Markdown (.md).
- Link that file in `./source/_toctree.yml` on the correct toc-tree.

Make sure to put your new file under the proper section. If you have a doubt, feel free to ask in a Github Issue or PR.

### Modifying FAQs

#### Adding new blocks or modifying existing blocks in existing sections
The FAQ blocks are automatically generated from the content of `./faq` folder using `../scripts/generate_faq.py`.  
To modify or add a question:
1. add or modify the question, tags, answer_file and source in the corresponding .json file located in the ./folder
2. create the answer in the corresponding sub-folder `./answers` as a .mdx file, named as the answer_file provided with the question, or modify the content of the corresponding answer_file

#### Adding new sections

You can add new sections to the FAQS by creating new folders under `./faq`.

The tree structure should look like:
```bash
docs/
├── faq
│   ├── section1_name
│   │   ├── answers
│   │   │   ├── sub_section1_name.mdx
│   │   │   ├── sub_section2_name.mdx
│   │   ├── sub_section1_name.json
│   │   ├── sub_section2_name.json
│   ├── section2_name
│   │   ├── answers
│   │   │   ├── sub_section1_name.mdx
│   │   ├── sub_section1_name.json
```

Add a placeholder for your new section by adding the markers:
```
<!-- FAQ:section1_name:sub_section1_name:start -->

<!-- FAQ:section1_name:sub_section1_name:end -->
```

All the elements located between these markers will be automatically modified with the questions contained in `./faq/section1_name/sub_section1_name.json`.

#### Showing specific FAQ blocks based on tags

You can insert FAQ blocks with specific tags in the documentation by adding the markers (example for the ASSEMBLY tag):
```
<!-- FAQ-TAGS:ASSEMBLY:start -->

<!-- FAQ-TAGS:ASSEMBLY:end -->
```

All the questions tagged `ASSEMBLY` will be inserted between these markers, using `../scripts/inject_faq_tags.py`.

#### Readiness

To improve readiness of the faqs when writing documentation, you can clean the automatically added content with: 
`../scripts/clean_faq_blocks.py`.  
This will remove both the content generated by section and by tag.

### Writing source documentation

Values that should be put in `code` should either be surrounded by backticks: \`like so\`. Note that argument names
and objects like True, None or any strings should usually be put in `code`.

#### Writing a multi-line code block

Multi-line code blocks can be useful for displaying examples. They are done between two lines of three backticks as usual in Markdown:

````
```
# first line of code
# second line
# etc
```
````

#### Adding an image

Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos, and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
the ones hosted on [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) in which to place these files and reference
them by URL. We recommend putting them in the following dataset: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
If an external contribution, feel free to add the images to your PR and ask a Hugging Face member to migrate your images
to this dataset.
