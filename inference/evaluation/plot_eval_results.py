import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
import pandas as pd


def restructure_metric_column_source(eval_dict, source_list, column_list, metric_list_detail):
    new_dict = {}
    for metric in metric_list_detail:
        name_dict = {}
        for name in column_list:
            num_dict = {}
            for num in source_list:
                num_dict[num] = eval_dict[name][num][metric]
            name_dict[name] = num_dict
        new_dict[metric] = name_dict

    return new_dict


def restructure_metric_source_column(eval_dict, source_list, column_list, metric_list_detail):
    new_dict = {}
    for metric in metric_list_detail:
        num_dict = {}
        for num in source_list:
            name_dict = {}
            for name in column_list:
                name_dict[name] = eval_dict[name][num][metric]
            num_dict[num] = name_dict
        new_dict[metric] = num_dict

    return new_dict


def plot_bar_groupby_source(
        dict_single_metric, 
        metric_name=None,
        trans_types=None, 
        src_types=None, 
        ylim=None,
        show_chart=True, 
        save_path=None
    ):
    """
    Plot a grouped bar chart for translation scores grouped by dataset.

    Parameters:
    - dict_single_metric (dict): Dictionary containing translation scores (of a single metric type) for each dataset, translator type.
    - trans_types (list, optional): List of translator types to include in the plot. Default is None (include all).
    - src_types (list, optional): List of dataset IDs to include in the plot. Default is None (include all).

    Output:
    - Displayed bar plot in the console.
    """
    if trans_types is None:
        trans_types = list(dict_single_metric.keys())
    if src_types is None:
        src_types = list(dict_single_metric[trans_types[0]].keys())
    id_to_name = {
        111: '전문분야',
        124: '기술과학1',
        125: '사회과학',
        126: '일반',
        563: '산업정보(특허)',
        71265: '일상생활 및 구어체',
        71266: '기술과학2',
        71382: '방송콘텐츠'
    }

    # Korean font
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font='NanumGothic')

    # Reshape dict_single_metric for seaborn
    if metric_name is None:
        metric_name = 'Translation Metric'
    reshaped_data = {'Dataset': [], metric_name: [], 'Translator Type': []}
    for trans_type in trans_types:
        for id, score in dict_single_metric[trans_type].items():
            if id not in src_types:
                continue
            reshaped_data['Dataset'].append(id_to_name[id])
            reshaped_data[metric_name].append(score)
            reshaped_data['Translator Type'].append(trans_type[:-6])

    # Create a DataFrame from reshaped dict_single_metric
    df = pd.DataFrame(reshaped_data)

    # Use seaborn to create a grouped bar plot
    plt.figure(figsize=(15, 8))  # Increase figure size
    sns.barplot(x='Dataset', y=metric_name, hue='Translator Type', data=df)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(metric_name + ' Scores Grouped by Dataset')

    # Move legend to the upper-right corner
    plt.legend(loc='upper right')

    if show_chart:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()


def plot_bar_groupby_translator(
        dict_single_metric, 
        metric_name=None,
        trans_types=None, 
        src_types=None, 
        ylim=None,
        show_chart=True, 
        save_path=None
    ):
    """
    Plot a grouped bar chart for translation scores grouped by translator.

    Parameters:
    - dict_single_metric (dict): Dictionary containing translation scores (of a single metric type) for each dataset, translator type.
    - trans_types (list, optional): List of translator types to include in the plot. Default is None (include all).
    - src_types (list, optional): List of dataset IDs to include in the plot. Default is None (include all).

    Output:
    - Displayed bar plot in the console.
    """
    if trans_types is None:
        trans_types = list(dict_single_metric.keys())
    if src_types is None:
        src_types = list(dict_single_metric[trans_types[0]].keys())
    id_to_name = {
        111: '전문분야',
        124: '기술과학1',
        125: '사회과학',
        126: '일반',
        563: '산업정보(특허)',
        71265: '일상생활 및 구어체',
        71266: '기술과학2',
        71382: '방송콘텐츠'
    }

    # Korean font
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font='NanumGothic')

    # Reshape dict_single_metric for seaborn
    if metric_name is None:
        metric_name = 'Translation Metric'
    reshaped_data = {'Dataset': [], metric_name: [], 'Translator Type': []}
    for trans_type in trans_types:
        for id, score in dict_single_metric[trans_type].items():
            if id not in src_types:
                continue
            reshaped_data['Dataset'].append(id_to_name[id])
            reshaped_data[metric_name].append(score)
            reshaped_data['Translator Type'].append(trans_type[:-6])

    # Create a DataFrame from reshaped dict_single_metric
    df = pd.DataFrame(reshaped_data)

    # Use seaborn to create a grouped bar plot
    plt.figure(figsize=(15, 8))  # Increase figure size
    rainbow_palette = sns.color_palette("rainbow", n_colors=len(src_types))
    sns.barplot(x='Translator Type', y=metric_name, hue='Dataset', data=df, palette=rainbow_palette)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(metric_name + ' Scores Grouped by Translator')

    # Move legend to the upper-right corner
    plt.legend(loc='upper right')

    if show_chart:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()


if __name__ == '__main__':
    import yaml

    source_list = [111, 124, 125, 126, 563, 71265, 71266, 71382]
    column_list = [
        'google_trans', 
        'deepl_trans', 
        'mbart_trans', 
        'nllb-600m_trans', 
        'nllb-1.3b_trans',
        'madlad_trans'
    ]
    metric_list_detail = ['bleu', 'sacrebleu', 'rouge_1', 'rouge_2', 'wer']

    yaml_path = '../results/test_tiny_uniform100_metrics_by_translator.yaml'
    with open(yaml_path, 'r') as file:
        result_by_translator = yaml.safe_load(file)

    dict_metric_column_source = restructure_metric_column_source(result_by_translator, source_list, column_list, metric_list_detail)
    for metric in metric_list_detail:
        if metric == 'bleu':
            ylim = (0, 70)
            metric_name = 'BLEU'
        elif metric == 'sacrebleu':
            ylim = (0, 50)
            metric_name = 'SacreBLEU'
        elif metric == 'rouge_1':
            ylim = (0, 50)
            metric_name = 'Rouge-1'
        elif metric == 'rouge_2':
            ylim = (0, 50)
            metric_name = 'Rouge-2'
        elif metric == 'wer':
            ylim = (0, 80)
            metric_name = 'WER'

        save_path_groupby_source = '../results/eval_results_plot_images/' + metric + '_groupby_dataset.png'
        plot_bar_groupby_source(
            dict_metric_column_source[metric],
            metric_name=metric_name,
            trans_types=None,
            src_types=None,
            ylim=ylim,
            show_chart=False,
            save_path=save_path_groupby_source
        )

        save_path_groupby_translator = '../results/eval_results_plot_images/' + metric + '_groupby_translator.png'
        plot_bar_groupby_translator(
            dict_metric_column_source[metric],
            metric_name=metric_name,
            trans_types=None,
            src_types=None,
            ylim=ylim,
            show_chart=False,
            save_path=save_path_groupby_translator
        )
