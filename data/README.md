# data/

Git does not track the files in this folder, except this README. The pipeline needs two public
review files. Download them yourself and put them here. Do not commit them.

| File | Source | Size | Columns that crossrec reads |
|---|---|---|---|
| `Reviews.csv` | Amazon Fine Food Reviews (Stanford SNAP), <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews> | about 300 MB, about 568,000 reviews | `UserId`, `ProductId`, `Score`, `Time` |
| `Books_rating.csv` | Amazon Books Reviews, <https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews> | about 2.9 GB, about 3,000,000 reviews | `User_id`, `Id`, `review/score`, `review/time`, `Title` |

## Terms

- Each dataset has its own license and terms on its Kaggle page. Read them before you use the data.
- The Fine Food data comes from the SNAP project (McAuley and Leskovec, 2013). Cite it if you publish results.
- The reviews contain reviewer profile names and review text. crossrec does not read these columns.
  Do not publish them again.

## Download

```bash
pip install kaggle                     # needs a Kaggle API token in ~/.kaggle/kaggle.json
kaggle datasets download -d snap/amazon-fine-food-reviews -f Reviews.csv -p data --unzip
kaggle datasets download -d mohamedbakhet/amazon-books-reviews -f Books_rating.csv -p data --unzip
```

You can also download the files from the two web pages.

## Expected result

```
data/
├── README.md
├── Reviews.csv
└── Books_rating.csv
```

Then run `crossrec stats --data-dir data`.

## No download

`crossrec synth --out data/synthetic` writes two synthetic files with the same column names.
The `--synthetic` flag of each command makes the same data in memory. The tests use only this
synthetic data.
