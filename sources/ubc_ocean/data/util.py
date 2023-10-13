import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_folds(source: str, target: str, seed=42):
    df = pd.read_csv(source)
    df['is_tma'] = df['is_tma'].astype('int').astype('str')
    df['stratify'] = df['label'] + df['is_tma']
    print(df.head())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = skf.split(df, df['stratify'])
    for fold, (train_indices, valid_indices) in enumerate(folds, 1):
        df.loc[valid_indices, 'fold'] = fold
    df['fold'] = df['fold'].astype(int)
    df.drop(columns=['stratify'], inplace=True)
    df.to_csv(target, index=False)


def main():
    src_df = 'data/original/annotations/train.csv'
    dst_df = 'data/original/annotations/train_folded.csv'
    create_folds(src_df, dst_df)


if __name__ == '__main__':
    df = pd.read_csv('data/original/annotations/train_folded.csv')
    for i in range(1, 6):
        print(i, df[df['fold'] == i]['label'].value_counts().to_dict())
        print(i, len(df[df['is_tma'] == 0]))
        # print(i)
    # print(df['label'].value_counts().to_dict())
    # print(df['stratify'].value_counts().to_dict())
    #
    # main()
