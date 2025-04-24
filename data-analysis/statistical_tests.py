import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# statistics
from scipy.stats import shapiro, kstest
from scipy.stats import levene
from scipy.stats import f_oneway
from statsmodels.stats.anova import anova_lm
from scipy.stats import boxcox
from scipy.stats import yeojohnson
from pingouin import welch_anova as pg_welch_anova
from scipy import stats
import scikit_posthocs as sp
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from load_data import LoadData

def shapiro_wilk(df, column_name, key):
    print("Starting Shapiro-Wilk test...")
    all_normal = True
    grouped = df

    num_groups = 0

    for group_name, group_data in grouped:
        data_to_test = group_data[column_name]
        
        if len(data_to_test) >= 3:
            stat, p_value = shapiro(data_to_test)
            
            print(f"Group: {group_name}")
            print(f"Shapiro-Wilk statistic: {stat}")
            print(f"P-value: {p_value}")
            
            if p_value > 0.05:
                print("The data is likely normally distributed (fail to reject H0).\n")
            else:
                all_normal = False
                print("The data is likely not normally distributed (reject H0).\n")
        else:
            print(f"Group {group_name} does not have enough data to run the Shapiro-Wilk test.\n")
        num_groups+=1
    if not all_normal:
        print("Shapiro-Wilk test indicates non-normal distribution among groups. Running Welch's ANOVA and Kruskal-Wallis...")
        welch_stat, welch_p = welch_anova(df, column_name)
        if num_groups >= 2:
            kruskal_wallis_test(df, column_name)
        else:
            print("Kruskal requires at least two groups.")
        print(f"Welch's ANOVA statistic: {welch_stat}, p-value: {welch_p}")
    return all_normal

def levene_test(grouped_df, value):
    try:
        group_values = [group[value].values for _, group in grouped_df]
        stat, p = levene(*group_values)
        print(f"Levene's test statistic: {stat}, p-value: {p}")
        if p < 0.05:
            print("Levene's test indicates unequal variances. Running Welch's ANOVA and Kruskal-Wallis...")
            welch_stat, welch_p = welch_anova(grouped_df, value)
            kruskal_wallis_test(grouped_df, value)
            print(f"Welch's ANOVA statistic: {welch_stat}, p-value: {welch_p}")
        else:
            # parametric
            print("Levene's test indicates equal variances. Running ANOVA...")
            one_way_anova(grouped_df)
    except Exception as e:
        print(f'Error running levene test, ', e)


def one_way_anova(grouped_df):
    groups = [group['avg_energy_pct'] for _, group in grouped_df]
    stat, p_value = f_oneway(*groups)
    print(f"ANOVA test: p = {p_value}")
    if p_value < 0.05:
         print(f"ANOVA test rejects the null hypothesis. There is statistical difference among groups.)")
         print("Since ANOVA is significant we perform Tukey's HSD test...")
         tukey_hsd(grouped_df)
    else:
        print(f"ANOVA test fails to reject the null hypothesis. There is not statistical difference among groups.)")

def welch_anova(grouped_df, value):
    group_values = [group[value].values for _, group in grouped_df]
    k = len(group_values)
    means = [group.mean() for group in group_values]
    variances = [group.var(ddof=1) for group in group_values]
    ns = [len(group) for group in group_values]
    print(f"Group means: {means}")
    print(f"Group variances: {variances}")
    print(f"Group sizes: {ns}")
    numerator = sum((mean - sum(means)/k) ** 2 / var for mean, var in zip(means, variances))
    denominator = sum(var / n for var, n in zip(variances, ns))
    print(f"Numerator: {numerator}")
    print(f"Denominator: {denominator}")
    welch_stat = numerator / denominator
    df = sum(1 / (n - 1) for n in ns)
    print(f"Degrees of freedom: {df}")
    welch_p = 1 - stats.chi2.cdf(welch_stat, df)
    return welch_stat, welch_p

def kruskal_wallis_test(grouped_df, value):
    group_values = [group[value].values for _, group in grouped_df]
    stat, p = stats.kruskal(*group_values)
    print(f"Kruskal-Wallis H statistic: {stat}, p-value: {p}")
    
    if p < 0.05:
        print("The test indicates a significant difference between the groups. Running post-hoc Dunn's test")
        dunn_test(grouped_df, value)
    else:
        print("The test indicates no significant difference between the groups.")

def dunn_test(grouped_df, value):
    group_values = [group[value].values for _, group in grouped_df]
    all_values = []
    group_labels = []
    for label, group in grouped_df:
        all_values.extend(group[value].values)
        group_labels.extend([label] * len(group))
    data = pd.DataFrame({
        'values': all_values,
        'group': group_labels
    })
    dunn_result = sp.posthoc_dunn(data, val_col='values', group_col='group', p_adjust='bonferroni')
    print("Dunn's test results (adjusted p-values):")
    print(dunn_result)
    return dunn_result

def tukey_hsd(grouped_df):
    values = []
    labels = []
    for group_name, group in grouped_df:
        values.extend(group['avg_energy_pct'])
        labels.extend([group_name] * len(group))
    tukey_result = pairwise_tukeyhsd(values, labels, alpha=0.05)
    print("\nTukey's HSD Test Results:")
    print(tukey_result)
  
def gen_single_density_graph(df, filter: list):
    for (keys), group in df:
        sns.histplot(
            group['avg_energy_pct'], 
            bins=10, 
            kde=True, 
            stat="density",
            label=f"{keys}", 
            alpha=0.5
        )
    plt.xlabel("Power (W)")
    plt.ylabel("Density")
    plt.legend()

    variables = [algo, component, "density_energy", "_".join(filter)]

    # PDF
    output_file_name = "_".join(str(var) for var in variables) + ".pdf"
    output_path = os.path.join(d_folder, output_file_name)
    print(f"Saving image {output_path}")
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()

def gen_boxplot_graph(df, filter):
    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df, x='msg_type', y='avg_energy_pct', hue='cli')
    plt.xlabel("")
    plt.ylabel("Power (W)")
    plt.tight_layout()
    plt.xticks(rotation=90)

    variables = [algo, component, "energy_boxplot", "_".join(filter)]

    # PDF
    output_file_name = "_".join(str(var) for var in variables) + ".pdf"
    output_path = os.path.join(d_folder, output_file_name)
    print(f"Saving image {output_path}")
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()

def statistical_tests():
    global component
    global algo
    global d_folder

    components = {'server', 'client'}
    algos = {'pubsub'}
    intervals = {0.1,0.2,0.5,1.0}
    cli = {1,2}
    msg_size = {1,2,3}
    msg_types = {
            "Image", "Pose", "JointState", "PointCloud2",
            "Imu", "String", "PoseStamped", "LaserScan",
            "Float64", "Float32", "Vector3Stamped", "Float64MultiArray",
            "TwistStamped", "PoseWithCovarianceStamped", "Int32"
            }
    transformations = {False}
    list_outliers = {False}

    for algo in algos:
        algo_folder=''
        dest_folder=''
        num_rows=0

        match algo:
            case 'pubsub':
                algo_folder='message_types-stdalone-1'
                dest_folder='pubsub'
                num_rows=2761

        # Paths
        s_folder = f"../exp_runners/experiments/{algo_folder}"
        root_d_folder = f"./graphs/{dest_folder}"
        os.makedirs(root_d_folder, exist_ok=True)

        # Load Data Class
        l_data = LoadData(num_rows, s_folder, algo)

        transf_label = 'no_transf'
        out_label = 'no_out'

        print('#### Publisher and Subscriber ####')

        # Diff Server and Client
        df_server = l_data.load_power('server', False, False)
        df_client = l_data.load_power('client', False, False)
        df_server['source'] = 'server'
        df_client['source'] = 'client'
        # Combine
        combined = pd.concat([df_server, df_client], ignore_index=True)

        # Group by source
        grouped_by_source = combined.groupby('source')

        shapiro_wilk(grouped_by_source, 'avg_energy_pct', 'source')

        print('### Number of Clients')
        msgs_to_filter = ["LaserScan"]
        filtered_df = df_client[df_client['msg_type'].isin(msgs_to_filter)]
        grouped_by_source = filtered_df.groupby('cli')
        shapiro_wilk(grouped_by_source, 'avg_energy_pct', 'cli')

        print('### Low Energy Msgs: Client ###')
        msgs_to_filter = ["Pose", "JointState", "Float64",
            "Imu", "String", "PoseStamped", "Float32",
              "Vector3Stamped", "Float64MultiArray", "TwistStamped",
             "PoseWithCovarianceStamped", "Int32"]
        filtered_df = df_client[df_client['msg_type'].isin(msgs_to_filter)]
        grouped_by_source = filtered_df.groupby('msg_type')
        shapiro_wilk(grouped_by_source, 'avg_energy_pct', 'source')

        print('### High Energy Msgs: Client ###')
        msgs_to_filter = ["Image", "LaserScan", "PointCloud2"]
        filtered_df = df_client[df_client['msg_type'].isin(msgs_to_filter)]
        grouped_by_source = filtered_df.groupby('msg_type')
        shapiro_wilk(grouped_by_source, 'avg_energy_pct', 'source')

        print('### Low Energy Msgs: Server ###')
        msgs_to_filter = ["Pose", "JointState", "Float64",
            "Imu", "String", "PoseStamped", "Float32",
              "Vector3Stamped", "Float64MultiArray", "TwistStamped",
             "PoseWithCovarianceStamped", "Int32"]
        filtered_df = df_server[df_server['msg_type'].isin(msgs_to_filter)]
        grouped_by_source = filtered_df.groupby('msg_type')
        shapiro_wilk(grouped_by_source, 'avg_energy_pct', 'source')

        print('### High Energy Msgs: Server ###')
        msgs_to_filter = ["Image", "LaserScan", "PointCloud2"]
        filtered_df = df_server[df_server['msg_type'].isin(msgs_to_filter)]
        grouped_by_source = filtered_df.groupby('msg_type')
        shapiro_wilk(grouped_by_source, 'avg_energy_pct', 'source')

        for transformation in transformations:
            if transformation:
                    transf_label = 'transf'
            for outliers in list_outliers:
                if outliers:
                    out_label = 'out'
                for component in components:
                    print(f"\n######################## Algorithm: {algo}, Component: {component}, Transformation: {transformation}, Outliers: {outliers} ########################\n")
                    d_folder = f"{root_d_folder}/{component}/{transf_label}-{out_label}"
                    os.makedirs(d_folder, exist_ok=True)
                    df_load = l_data.load_power(component, transformation, outliers)
                    print(f"Component: {component} --->")
                    grouped_df = l_data.group_df_by(df_load,'msg_type',outliers)
                    if shapiro_wilk(grouped_df, 'avg_energy_pct', 'msg_type'):
                        print("Levene's test for different msg_types.")
                        levene_test(grouped_df,'avg_energy_pct')
                        # gen_single_density_graph(grouped_df,{'py_and_cpp'})
                    for mtype in msg_types:
                        df = df_load
                        df_mtype = l_data.filter_df(df,'msg_type',mtype)
                        print(f"Component: {component}, Msg type: {mtype} --->")
                        cleaned_df = l_data.group_df_by(df_mtype,'msg_type',outliers)
                        for group_name, group_data in cleaned_df:
                            print(f"Group: {group_name}")
                            print(group_data['avg_energy_pct'].to_numpy())
                            # gen_single_density_graph(cleaned_df, {str(mtype)})
                        shapiro_wilk(cleaned_df, 'avg_energy_pct', 'MType')
                        print(f"## Message Size:  {component}, {mtype} ##")
                        grouped_df = l_data.group_df_by(df_mtype,'msg_size',outliers)
                        if shapiro_wilk(grouped_df, 'avg_energy_pct', 'msg_size'):
                            print(f"Levene's test for MType {mtype} and different msg_intervals.")
                            levene_test(grouped_df,'avg_energy_pct')
                        #     gen_single_density_graph(grouped_df,{mtype,'intervals'})
                        for s in msg_size:
                            df = df_mtype
                            df = l_data.filter_df(df,'msg_size',s)
                            df_interval = df
                            print(f"Component: {component}, MType: {mtype}, Size: {s},  --->")
                            cleaned_df = l_data.group_df_by(df,'msg_size', outliers)
                            for group_name, group_data in cleaned_df:
                                print(f"Group: {group_name}")
                                print(group_data['avg_energy_pct'].to_numpy())
                                # gen_single_density_graph(cleaned_df, {str(mtype),str(s)})
                            shapiro_wilk(cleaned_df, 'avg_energy_pct', 'msg_size') 
                        grouped_df = l_data.group_df_by(df_mtype,'msg_interval',outliers)
                        if shapiro_wilk(grouped_df, 'avg_energy_pct', 'msg_interval'):
                            print(f"Levene's test for MType {mtype} and different msg_intervals.")
                            levene_test(grouped_df,'avg_energy_pct')
                            # gen_single_density_graph(grouped_df,{mtype,'intervals'})   
                        for interval in intervals:
                            df = df_mtype
                            df = l_data.filter_df(df,'msg_interval',interval)
                            df_interval = df
                            print(f"Component: {component}, MType: {mtype}, Interval: {interval},  --->")
                            cleaned_df = l_data.group_df_by(df,'msg_interval', outliers)
                            for group_name, group_data in cleaned_df:
                                print(f"Group: {group_name}")
                                print(group_data['avg_energy_pct'].to_numpy())
                                # gen_single_density_graph(cleaned_df, {str(mtype),str(interval)})
                            shapiro_wilk(cleaned_df, 'avg_energy_pct', 'msg_interval')
                            grouped_df = l_data.group_df_by(df_interval,'cli',outliers)
                            if shapiro_wilk(grouped_df,'avg_energy_pct','cli'):
                                print(f"Levene's test for MType {mtype}, msg_interval {interval} and different cli.")
                                levene_test(grouped_df,'avg_energy_pct')
                                # gen_single_density_graph(grouped_df,{str(mtype),str(interval),'clients'})
                            for clients in cli:
                                df = df_interval
                                df = l_data.filter_df(df,'cli',clients)
                                print(f"Component: {component}, MType: {mtype}, Interval: {interval}, Clients: {clients} --->")
                                cleaned_df = l_data.group_df_by(df,'cli', outliers)
                                for group_name, group_data in cleaned_df:
                                    print(f"Group: {group_name}")
                                    print(group_data['avg_energy_pct'].to_numpy())
                                    # gen_single_density_graph(cleaned_df, {str(mtype),str(interval), str(clients)})
                                shapiro_wilk(cleaned_df, 'avg_energy_pct', 'cli')
                        grouped_df = l_data.group_df_by(df_mtype,'cli',outliers)
                        print(f"MType {mtype} and different cli")
                        if shapiro_wilk(grouped_df, 'avg_energy_pct', 'cli'):
                            print(f"Levene's test for MType {mtype} and different cli.")
                            levene_test(grouped_df,'avg_energy_pct')
                            # gen_single_density_graph(grouped_df,{mtype,'clients'})
                        for clients in cli:
                                df = df_mtype
                                df = l_data.filter_df(df,'cli',clients)
                                print(f"Component: {component}, Clients: {clients}, MType: {mtype} --->")
                                cleaned_df = l_data.group_df_by(df,'cli', outliers)
                                for group_name, group_data in cleaned_df:
                                    print(f"Group: {group_name}")
                                    print(group_data['avg_energy_pct'].to_numpy())
                                    # gen_single_density_graph(cleaned_df, {str(mtype), str(clients)})
                                shapiro_wilk(cleaned_df, 'avg_energy_pct', 'cli')
                                # 1 client and different msg intervals
                                if clients == 1:
                                    grouped_df = l_data.group_df_by(df,'msg_interval',outliers)
                                    if shapiro_wilk(grouped_df,'avg_energy_pct','msg_interval'):
                                        print(f"Levene's test for MType {mtype}, cli {clients} and different msg_intervals.")
                                        levene_test(grouped_df,'avg_energy_pct')
                                        # gen_single_density_graph(grouped_df,{str(mtype),str(clients),'interval'})
                    if shapiro_wilk(grouped_df, 'avg_energy_pct', 'cli'):
                        print(f"Levene's test different cli.")
                        grouped_df = l_data.group_df_by(df_load,'cli',outliers)
                        levene_test(grouped_df,'avg_energy_pct')
                        # gen_single_density_graph(grouped_df,{'clients'})
                    for clients in cli:
                        df = df_load
                        df_clients = l_data.filter_df(df,'cli',clients)
                        print(f"Component: {component}, Num clients: {clients} --->")
                        cleaned_df = l_data.group_df_by(df_clients,'cli',outliers)
                        for group_name, group_data in cleaned_df:
                            print(f"Group: {group_name}")
                            print(group_data['avg_energy_pct'].to_numpy())
                            # gen_single_density_graph(cleaned_df, {str(cli)})
                        shapiro_wilk(cleaned_df, 'avg_energy_pct', 'cli')

                    grouped_df = l_data.group_df_by(df_load,'cli',outliers)
                    if shapiro_wilk(grouped_df,'avg_energy_pct','msg_interval'):
                        print(f"Levene's different msg_intervals.")
                        levene_test(grouped_df,'avg_energy_pct')
                        # gen_single_density_graph(grouped_df,{'intervals'})
                    for interval in intervals:
                        df = df_load
                        df_intervals = l_data.filter_df(df,'msg_interval',interval)
                        print(f"Component: {component}, Interval: {interval} --->")
                        cleaned_df = l_data.group_df_by(df_intervals,'msg_interval',outliers)
                        for group_name, group_data in cleaned_df:
                            print(f"Group: {group_name}")
                            print(group_data['avg_energy_pct'].to_numpy())
                            # gen_single_density_graph(cleaned_df, {str(interval)})
                        shapiro_wilk(cleaned_df, 'avg_energy_pct', 'msg_interval')
                        # gen_boxplot_graph(df_intervals,{str(interval)})
                        
                    time.sleep(5)

if __name__ == "__main__":
    statistical_tests()
            