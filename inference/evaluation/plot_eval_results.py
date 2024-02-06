import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
import pandas as pd


def restructure_metric_column(eval_dict, column_list, metric_list_detail):
    new_dict = dict()
    for metric in metric_list_detail:
        metric_dict = dict()
        for column in column_list:
            metric_dict[column] = eval_dict[column][metric]
        new_dict[metric] = metric_dict

    return new_dict


def restructure_metric_column_source(eval_dict, source_list, column_list, metric_list_detail):
    new_dict = {}
    for metric in metric_list_detail:
        name_dict = {}
        for name in column_list:
            num_dict = {}
            for num in source_list:
                num_dict[num] = eval_dict[num][name][metric]
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
                name_dict[name] = eval_dict[num][name][metric]
            num_dict[num] = name_dict
        new_dict[metric] = num_dict

    return new_dict


def plot_bar(
        eval_dict, 
        metric_name, 
        # trans_types=None, 
        ylim=None, 
        show_chart=True, 
        save_path=None
    ):
    # if trans_types is None:
    #     trans_types = list(eval_dict.keys())

    df = pd.DataFrame(list(eval_dict.items()), columns=['Translator', metric_name])
    df['Translator'] = df['Translator'].apply(lambda x: x.replace('_processed', ''))
    df['Translator'] = df['Translator'].apply(lambda x: x.replace('_trans', ''))

    plt.figure(figsize=(15, 8))
    sns.barplot(x='Translator', y=metric_name, data=df, palette='rainbow')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(metric_name + ' Scores')

    if show_chart:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()


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
            trans_type = trans_type.replace('_processed', '')
            trans_type = trans_type.replace('_trans', '')
            reshaped_data['Translator Type'].append(trans_type)

    # Create a DataFrame from reshaped dict_single_metric
    df = pd.DataFrame(reshaped_data)

    # Use seaborn to create a grouped bar plot
    plt.figure(figsize=(15, 8))
    palette = sns.color_palette("rainbow", n_colors=len(trans_types))
    sns.barplot(x='Dataset', y=metric_name, hue='Translator Type', data=df, palette=palette)
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
            trans_type = trans_type.replace('_processed', '')
            trans_type = trans_type.replace('_trans', '')
            reshaped_data['Translator Type'].append(trans_type)

    # Create a DataFrame from reshaped dict_single_metric
    df = pd.DataFrame(reshaped_data)

    # Use seaborn to create a grouped bar plot
    plt.figure(figsize=(15, 8))  # Increase figure size
    palette = sns.color_palette("viridis", n_colors=len(src_types))
    sns.barplot(x='Translator Type', y=metric_name, hue='Dataset', data=df, palette=palette)
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
        'papago_trans',
        'google_trans', 
        'deepl_trans', 
        'mbart_trans', 
        'nllb-600m_trans', 
        'madlad_trans',
        # 'llama_trans',
        'mbart-aihub_trans',
        # 'llama-aihub-qlora_trans'
        'llama-aihub-qlora_trans_processed',
        'llama-aihub-qlora-eos_trans_processed'
    ]
    metric_list_detail = ['sacrebleu'] # 'bleu', 'sacrebleu', 'rouge_1', 'rouge_2', 'wer'

    yaml_path_aihub = '../results/test_tiny_uniform100_metrics.yaml'
    yaml_path_flores = '../results/test_flores_metrics.yaml'
    with open(yaml_path_flores, 'r') as file:
        result = yaml.safe_load(file)
    
    yaml_path_by_source = '../results/test_tiny_uniform100_metrics_by_source.yaml'
    with open(yaml_path_by_source, 'r') as file:
        result_by_source = yaml.safe_load(file)

    dict_metric_column = restructure_metric_column(result, column_list, metric_list_detail)
    dict_metric_column_source = restructure_metric_column_source(result_by_source, source_list, column_list, metric_list_detail)
    for metric in metric_list_detail:
        if metric == 'bleu':
            metric_name = 'BLEU'
        elif metric == 'sacrebleu':
            metric_name = 'SacreBLEU'
        elif metric == 'rouge_1':
            metric_name = 'Rouge-1'
        elif metric == 'rouge_2':
            metric_name = 'Rouge-2'
        elif metric == 'wer':
            metric_name = 'WER'

        save_path_aihub = f'../results/chart_images/aihub_{metric}.png'
        save_path_flores = f'../results/chart_images/flores_{metric}.png'
        plot_bar(
            dict_metric_column[metric],
            metric_name=metric_name,
            ylim=(0, 30),
            show_chart=False,
            save_path=save_path_flores
        )

        # save_path_groupby_source = '../results/chart_images/aihub_' + metric + '_groupby_dataset.png'
        # plot_bar_groupby_source(
        #     dict_metric_column_source[metric],
        #     metric_name=metric_name,
        #     trans_types=None,
        #     src_types=None,
        #     # ylim=ylim,
        #     show_chart=False,
        #     save_path=save_path_groupby_source
        # )

        # save_path_groupby_translator = '../results/chart_images/aihub_' + metric + '_groupby_translator.png'
        # plot_bar_groupby_translator(
        #     dict_metric_column_source[metric],
        #     metric_name=metric_name,
        #     trans_types=None,
        #     src_types=None,
        #     # ylim=ylim,
        #     show_chart=False,
        #     save_path=save_path_groupby_translator
        # )

    def plot_speed(
            eval_speed, 
            ylim=(0,4), 
            show_chart=True, 
            save_path=None
        ):
        df = pd.DataFrame(list(eval_speed.items()), columns=['Translator', 'Speed'])
        df['Translator'] = df['Translator'].apply(lambda x: x.replace('_trans', ''))
        df['Translator'] = df['Translator'].apply(lambda x: x.replace('_processed', ''))
        
        plt.figure(figsize=(15, 8))
        sns.barplot(x='Translator', y='Speed', data=df, palette='rainbow')
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        plt.title('Inference Speed (sentence / sec)')

        if show_chart:
            plt.show()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
            plt.close()

    yaml_path_speed_aihub = '../results/test_tiny_uniform100_speeds.yaml'
    yaml_path_speed_flores = '../results/test_flores_speeds.yaml'
    with open(yaml_path_speed_aihub, 'r') as file:
        speed = yaml.safe_load(file)
    speed = {trans: speed[trans]['speed'] for trans in column_list}

    yaml_path_speed_aihub = '../results/chart_images/aihub_speed.png'
    yaml_path_speed_flores = '../results/chart_images/flores_speed.png'
    plot_speed(speed, show_chart=False, save_path=yaml_path_speed_aihub)


