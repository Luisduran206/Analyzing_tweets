# 🌐 Analyzing tweets

### _Objective:_

The objective of this practical work is to carry out an investigation on data sets extracted from `Twitter`, in order to discover relationships between different types of content data.
A number of insights were developed on the data. For more information, please browse the `Markdown` folder.
In this case, the tweets are related to the release of `The Avengers: End Game`.

The results will be stored in a `Markdown` file.

## 📂 **Requierements:**

- [python](https://www.python.org) >=3.10
- [uv](https://github.com/astral-sh/uv) as project manager
- [Black](https://black.readthedocs.io/en/stable/index.html) as a code formatter
- [Pylint](https://www.pylint.org/) as code linter
- [pandoc](https://pandoc.org/) as a markdown file convertion tool

## **User Guide:**

Once the repository has been cloned, it must be executed in the console:

```bash
uv run twitter.py
```

Once the code is executed, it will ask us for a data path which is: `./data/avengers_endgame.csv`

```bash
Path to CSV file (./data/avengers_endgame.csv): ./data/avengers_endgame.csv
```

After running the code, a `pandoc` folder will be generated where the images will be and the `markdown folder` that we will use to generate the report using the `pandoc tool.

```bash
mv ./pandoc/*.png ./markdown

cd markdown

pandoc introduccion.md metodologia.md analisis.md -o informe.pdf
```
