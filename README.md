# Steam Review aspect dataset

This dataset contains 1100 English Steam reviews, split into 900 train and 200 test. This dataset was initially used to identify which aspects are mentioned in English reviews as part of [Analysis of 43 million English Steam Reviews](https://srec.ai/blog/analysis-43m-english-steam-reviews). Most content on this README file originally appeared on SRec blog [Steam review aspect dataset](https://srec.ai/blog/steam-review-aspect-dataset).

## Data collection and annotation

The source of the reviews comes from a snapshot of the SRec database, which was taken on 21 February 2024. SRec obtain all reviews for all games and mods using [API provided by Steam](https://partner.steamgames.com/doc/store/getreviews). To reduce bias when selecting reviews to be annotated, I chose reviews primarily based on these criteria,

* Character length.
* Helpfulness score.
* Popularity of the reviewed game.
* Genre or category of the reviewed game.

There are 8 aspects to define review in this dataset. I am the only annotator for this dataset. A review is deemed to contain a certain aspect, even if it's mentioned implicitly (e.g., "but it'd be great if there's good looking characters...") or only mention lack of the aspect (e.g., "... essentially has no story ..."). The below table shows 8 aspects of this dataset, along with a short description and example.

> A description and example for 8 aspects in this dataset

| Aspect      | Short description                                                                                                     | Review text                                                                                                                    |
| ----------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Recommended | Whether the reviewer recommends the game or not. This aspect comes from the one who wrote the review.                 | ... In conclusion, good game                                                                                                   |
| Story       | Story, character, lore, world-building and other storytelling elements.                                               | Excellent game, but has an awful-abrupt ending that comes out of nowhere and doesnt make sense ...                             |
| Gameplay    | Controls, mechanics, interactivity, difficulty and other gameplay setups.                                             | Gone are the days of mindless building and fun. Power grids? Taxes? Intense accounting and counter-intuitive path building ... |
| Visual      | Aesthetic, art style, animation, visual effects and other visual elements.                                            | Gorgeous graphics + 80s/90s anime artstyle + Spooky + Atmospheric ...                                                          |
| Audio       | Sound design, music, voice acting and other auditory elements.                                                        | ... catchy music, wonderful narrator saying very kind words ...                                                                |
| Technical   | The technical aspects of the game such as bug, performance, OS support, controller support and overall functionality. | bad doesnt fit a 1080p monitor u bastard ...                                                                                   |
| Price       | Price of the game or its additional content.                                                                          | Devs are on meth pricing this game at $44                                                                                      |
| Suggestion  | Suggestions for the state of the game, including external factors such as game\'s price or publisher partnership.     | ... but needs a bit of personal effort to optimize the controls for PC, otherwise ...                                          |

Take note that few reviews contain language and content that some people may find offensive, discriminatory, or inappropriate. I **DO NOT** endorse, condone or promote any of such language and content.

## Data format

CSV, JSON and Apache Arrow file formats are provided for convenience's sake. You can check the notebook on `example` directory for a bare-minimum example of how to open those files. Both raw and cleaned review text are provided. Cleaned review text was preprocessed by stripping BBcode, reducing excessive whitespaces and reducing excessive newlines.

## Model benchmark

Click [here](./model_benchmark/README.md).

## Download

You can download Steam review aspect dataset from here (GitHub) or one of these sources,

* [Huggingface](https://huggingface.co/datasets/ilos-vigil/steam-review-aspect-dataset)
* [Kaggle](https://www.kaggle.com/datasets/ilosvigil/steam-review-aspect-dataset)

# Citation

If you wish to use this dataset in your research or project, please cite this blog post: [Steam review aspect dataset](https://srec.ai/blog/steam-review-aspect-dataset) 

```
Sandy Khosasi. "Steam review aspect dataset". (2024).
```

```bibtex
@misc{srec:steam-review-aspect-dataset,
	title        = {Steam review aspect dataset},
	author       = {Sandy Khosasi},
	year         = {2024},
	month        = {may},
	day          = {29},
	url          = {https://srec.ai/blog/steam-review-aspect-dataset},
    urldate      = {2024-05-29}
}
```

# License

Steam Review aspect dataset is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0).


## Statistic

> Total occurrence of each aspect

| Aspect      | Train | Test |
| ----------- | ----- | ---- |
| Recommended | 667   | 148  |
| Story       | 400   | 89   |
| Gameplay    | 693   | 154  |
| Visual      | 391   | 87   |
| Audio       | 227   | 51   |
| Technical   | 259   | 57   |
| Price       | 213   | 47   |
| Suggestion  | 97    | 21   |

> Total aspect in a review

| Total aspect | Train | Test |
| ------------ | ----- | ---- |
| 0            | 1     | 7    |
| 1            | 88    | 11   |
| 2            | 214   | 43   |
| 3            | 218   | 55   |
| 4            | 184   | 49   |
| 5            | 140   | 21   |
| 6            | 46    | 8    |
| 7            | 7     | 5    |
| 8            | 2     | 1    |

> Total review for each game in this dataset

| Total review for each game | Train | Test |
| -------------------------- | ----- | ---- |
| 1                          | 280   | 164  |
| 2                          | 301   | 18   |
| 3                          | 6     | 0    |

> Statistics of total characters

|             | Train (review) | Train (cleaned review) | Test (review) | Test (cleaned review) |
| ----------- | -------------- | ---------------------- | ------------- | --------------------- |
| Q1          | 417            | 416.75                 | 390           | 390                   |
| Q2 (Median) | 871            | 867.5                  | 888           | 888                   |
| Q3          | 1810.5         | 1753.75                | 1629.75       | 1623.5                |
| Average     | 1408.49        | 1389.06                | 1286.12       | 1267.96               |