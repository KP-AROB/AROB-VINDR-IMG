import pandas as pd
import os
import ast


def format_char(char):
    return char.lower().replace(' ', '_')


def format_category_list(category_list):
    return [format_char(category) for category in category_list]


def contains_all_classes(category_list, class_list):
    return any(cls in category_list for cls in class_list)


def replace_categories(df, column, target_categories):
    def replace_if_present(categories):
        for target in target_categories:
            if target in categories:
                return target
        return categories

    df[column] = df[column].apply(
        lambda x: replace_if_present(x) if isinstance(x, list) else x)
    return df


def prepare_vindr_finding_dataframe(data_dir, class_list, train: bool = True):
    df_find = pd.read_csv(os.path.join(
        data_dir, 'finding_annotations.csv'))
    df_find['finding_categories'] = df_find['finding_categories'].apply(
        ast.literal_eval)
    df_find['finding_categories'] = df_find['finding_categories'].apply(
        format_category_list)
    df_find = df_find[df_find['finding_categories'].apply(
        lambda x: contains_all_classes(x, class_list))]
    df_find = replace_categories(df_find, 'finding_categories', class_list)
    df_find.drop_duplicates(subset='image_id', keep=False, inplace=True)
    split_name = 'training' if train else 'test'
    df_find = df_find[df_find['split'] == split_name]
    return df_find


def prepare_vindr_birads_dataframe(data_dir, train: bool = True):

    df_find = pd.read_csv(os.path.join(
        data_dir, 'finding_annotations.csv'))
    df_find['breast_birads'] = df_find['breast_birads'].apply(format_char)

    mapping = {
        'bi-rads_1': 'bi-rads_1',
        'bi-rads_2': 'bi-rads_2',
        'bi-rads_3': 'bi-rads_0',
        'bi-rads_4': 'bi-rads_0',
        'bi-rads_5': 'bi-rads_0'
    }

    df_find['breast_birads'] = df_find['breast_birads'].replace(mapping)
    split_name = 'training' if train else 'test'
    df_find = df_find[df_find['split'] == split_name]
    return df_find
