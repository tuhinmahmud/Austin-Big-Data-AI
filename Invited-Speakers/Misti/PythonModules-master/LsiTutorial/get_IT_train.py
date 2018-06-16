import pandas as pd

def extract_category(infile, category, outfile):
    train = pd.read_csv(infile)
    it_train=train.ix[train.Category==category]
    it_train.FullDescription.to_csv(outfile, header=False, index=False)


if __name__ == "__main__":
    extract_category("data/Train_rev1.csv", "IT Jobs", "data/Train_IT.csv")
