# # Directories to save results
# # from tqdm import tqdm  # Import the tqdm library for progress bar
# # import time  # Import time for simulating or handling elapsed time calculation
# # import os
# # import numpy as np
# # import pandas as pd
# # import dask.array as da
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # from IPython.display import display, clear_output
# # import ipywidgets as widgets

# # import scipy as sp
# # from scipy.stats import chi2
# # from sklearn.covariance import MinCovDet
# # from sklearn.decomposition import PCA
# # from matplotlib.lines import Line2D
# # from sklearn.preprocessing import StandardScaler

# # Check and create 'data' and 'results' folders if they do not exist
# data_folder = 'data'
# results_folder = 'results'

# if not os.path.exists(data_folder):
#     os.makedirs(data_folder)
# if not os.path.exists(results_folder):
#     os.makedirs(results_folder)

# # Assign folder paths to variables for later use
# DATA_DIR = data_folder
# RESULTS_DIR = results_folder


# def init_results_dirs(base_dir: str = "results", event: str | None = None):
#     """
#     Initialize results directory structure with per-event subfolders.

#     Structure created:
#         results/
#           cleaned_new/ <event_dir>
#           variances/   <event_dir>
#           components/  <event_dir>
#           eigenvalues/ <event_dir>
#           biplots/     <event_dir>
#           cos_plots/   <event_dir>

#     Where <event_dir> is "event_<event>" if event is provided, otherwise "all_events".

#     Args:
#         base_dir: Base results directory.
#         event: Optional event name.

#     Returns:
#         dict: Mapping of category name to absolute event-specific directory path. Includes key "base".
#     """
#     base_abs = os.path.abspath(base_dir)
#     os.makedirs(base_abs, exist_ok=True)

#     event_dirname = f"event_{event}" if (event and len(str(event)) > 0) else "all_events"

#     categories = [
#         "cleaned_new",
#         "variances",
#         "components",
#         "eigenvalues",
#         "biplots",
#         "cos_plots",
#     ]

#     dirs: dict[str, str] = {"base": base_abs}
#     for cat in categories:
#         cat_parent = os.path.join(base_abs, cat)
#         os.makedirs(cat_parent, exist_ok=True)
#         cat_event_dir = os.path.join(cat_parent, event_dirname)
#         os.makedirs(cat_event_dir, exist_ok=True)
#         dirs[cat] = cat_event_dir

#     return dirs


#     # Robust Mahalonibis Distance
# def robust_mahalanobis_method_dask(df, md_name='',outlier_name='', p_md_name='', cut=0.001):
#     """
#     Calculate a robust version of Mahalanobis distances for each data subset,
#     using the Minimum Covariance Determinant (MCD) method.
#     Args:
#         df (pd.DataFrame): Dataframe to calculate Mahalanobis distances for.
#         md_name (str): Name of the Mahalanobis distance column.
#         outlier_name (str): Name of the outlier column.
#         p_md_name (str): Name of the probability of the Mahalanobis distance column.
#         cut (float): Cut-off point for the Mahalanobis distance.
#     Returns:
#         pd.DataFrame: DataFrame with Mahalanobis distances and outlier flags
#     """
#     #Minimum covariance determinant
#     rng = np.random.RandomState(0)
#     real_cov = np.cov(df.values.T)
#     # print(df.columns)
#     # print(real_cov)
#     # Check if the covariance matrix is all zeros
#     if np.all(real_cov == 0):
#         # Check the dimensions of real_cov
#         if real_cov.ndim == 0:  # Only one variable present
#         # Create a 1x1 perturbation matrix
#             perturbation = np.array(1e-2)  # Increase here
#         else:  # Multiple variables present
#             # Add a larger perturbation to the diagonal
#             perturbation = np.eye(real_cov.shape[0]) * 1e-2  # Increase here
#         real_cov += perturbation
#         # print(f'PERTURBATION!: {real_cov}')
#     # Calculate inverse covariance
#     if df.shape[1] < 2:
#         # Reshape when only one column in used (e.g., 'SteeringInput')
#         X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov.reshape((1,1)), 
#             size= round(len(df) * 0.5)) # 50% of the data
#         cov = MinCovDet(random_state=0,support_fraction=0.8).fit(X) #calculate covariance
#         mcd = cov.covariance_ #robust covariance metric
#         robust_mean = cov.location_  #robust mean
#         inv_covmat = sp.linalg.inv(mcd) #inverse of covariance matrix
#     else:
#         X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov, size= round(len(df) * 0.5))
#         cov = MinCovDet(random_state=0,support_fraction=0.8).fit(X) #calculate covariance
#         mcd = cov.covariance_ #robust covariance metric
#         robust_mean = cov.location_  #robust mean
#         inv_covmat = sp.linalg.inv(mcd) #inverse of covariance matrix

#     # Robust M-Distance
#     x_minus_mu = df - robust_mean
#     # Transform data into dask arrays
#     x_minus_mu_dask = da.from_array(x_minus_mu.to_numpy(), chunks=(min(x_minus_mu.shape[0], 10000), df.shape[1]))
#     mahal = da.sqrt(da.diagonal(da.dot(da.dot(x_minus_mu_dask,inv_covmat), x_minus_mu_dask.T)))
#     # Calculate md
#     md = mahal.compute()
#     # Compute the chi-squared cumulative probability distribution to transfer the md2 into probabilities
#     probability_md = 1 - chi2.cdf(md, df=df.shape[1])

#     # Save md values and probabilities to df column
#     md_df = pd.DataFrame({md_name:md,p_md_name:probability_md})
#     # Set a Chi2 cut-off point using probability of 0.01 (99.5% Chi2 quantile)
#     # Degrees of freedom (df) = number of variables
#     threshold = chi2.ppf((1-cut), df=df.shape[1])
#     # STD threshold, assuming 'md' contains your computed distances
#     # mean_md = np.mean(md)
#     # std_md = np.std(md)
#     # threshold_3std = mean_md + 4 * std_md
#     # Flag outliers as md > threshold
#     md_df[outlier_name] = md_df[md_name] > threshold
#     return md_df


# # Mahalanobis distance fucntion to be used in the PCA analysis
# def mahalanobis_for_pca(df):
#     """
#     Calculate Mahalanobis distances for each data subset.
#     Args:
#         df (pd.DataFrame): Input DataFrame containing the data to process
#     Returns:
#         pd.DataFrame: DataFrame with Mahalanobis distances and outlier flags 
#         for all set of variables
#     """
#     ## ---- EYE columns ----
#     mds_eye = robust_mahalanobis_method_dask(df=df[['eye_theta_h', 'eye_theta_v']], md_name='md_eye',
#     outlier_name='eye_outlier', p_md_name='p_md_eye', cut=0.10).reset_index(drop=True)
#     ## ---- HEAD columns ----
#     mds_head = robust_mahalanobis_method_dask(df=df[['RelativeHeadYaw_degrees','RelativeHeadPitch_degrees',
#      'RelativeHeadRoll_degrees']], md_name='md_head',outlier_name='head_outlier', p_md_name='p_md_head', cut=0.20).reset_index(drop=True)
#     ## ---- CAR columns ----
#     mds_car = robust_mahalanobis_method_dask(df=df[['CarYaw_degrees', 'CarPitch_degrees', 'CarRoll_degrees']],
#      md_name='md_car',outlier_name='car_outlier', p_md_name='p_md_car', cut=0.10).reset_index(drop=True)
#     ## ---- Steering ----
#     mds_steer = robust_mahalanobis_method_dask(df=df[['streeringDegree']], md_name='md_steering',
#     outlier_name='steering_outlier', p_md_name='p_md_steer', cut=0.10).reset_index(drop=True)

#     ## save md and outlier data
#     md_df = pd.concat([df.reset_index(drop=True),mds_eye, mds_head, mds_car,mds_steer], axis=1)
#     final_df = md_df.drop(columns=['md_eye','p_md_eye','md_head','p_md_head','md_car','p_md_car',
#     'md_steering','p_md_steer'])
#     return final_df

# # Function to interpolate NaN values in a subset of data
# def interpolate_nan_pca(df):
#     """
#     Interpolate NaN values in a subset of data.
#     Args:
#         df (pd.DataFrame): Input DataFrame containing the data to process
#     Returns:
#         pd.DataFrame: DataFrame with interpolated NaN values
#     """
#     # 1. Convert 'outlier' column values to NaN where True
#     # -- Eye
#     df.loc[df['eye_outlier'], ['eye_theta_h', 'eye_theta_v']] = np.nan

#     # -- Head
#     df.loc[df['head_outlier'], ['RelativeHeadYaw_degrees', 'RelativeHeadPitch_degrees','RelativeHeadRoll_degrees']] = np.nan
#     # -- Car
#     df.loc[df['car_outlier'], ['CarRoll_degrees','CarYaw_degrees', 'CarPitch_degrees']] = np.nan
#     # -- Steering
#     df.loc[df['steering_outlier'], ['streeringDegree']] = np.nan
#     # Group by 'uid' and interpolate NaN values
#     interpolated_df = df.groupby('conditions').apply(lambda group: group.interpolate(method='linear').ffill().bfill()).reset_index(drop=True)
#     # cleaned_df = interpolated_df.drop(columns=['eye_outlier','head_outlier','car_outlier','steering_outlier','pupil_dilation_outlier'])
#     return interpolated_df

# # Function to clean a subset of data by detecting and interpolating outliers during PCA analysis
# def clean_subset(df):
#     """
#     Clean a subset of data by detecting and interpolating outliers.
#     Args:
#         df (pd.DataFrame): Input DataFrame containing the data to process
#     Returns:
#         pd.DataFrame: DataFrame with interpolated NaN values
#     """
#     # Detect outliers in subset by md
#     subset_md = df.pipe(mahalanobis_for_pca)
#     # Interpolate detected outliers
#     subset_interpolated = subset_md.pipe(interpolate_nan_pca)
#     return subset_interpolated

# # Mapping of old labels to new labels
# def map_labels(labels, label_mapping):
#     """
#     Maps each label in the given list using the provided mapping dictionary.

#     Args:
#         labels (list): List of original labels.
#         label_mapping (dict): Mapping dictionary for label replacement.

#     Returns:
#         list: New list with mapped labels.
#     """
#     return [label_mapping.get(label, label) for label in labels]

# # Define your label mapping (you can modify this as needed)
# custom_mapping = {
#     'eye_theta_h': 'Eye Horizontal',
#     'eye_theta_v': 'Eye Vertical',
#     'RelativeHeadYaw_degrees': 'Head Yaw',
#     'RelativeHeadPitch_degrees': 'Head Pitch',
#     'RelativeHeadRoll_degrees': 'Head Roll',
#     'CarYaw_degrees': 'Car Yaw',
#     'streeringDegree': 'Steering',
#     'conditions': 'Condition',
# }

# # Function to plot a biplot of PCA scores and loadings
# def biplot(score, coef, eigenvalues,
#            labels=None,
#            colors=None,
#            explained_variance=None,
#            vector_colors=None,
#            scaled=None,
#            vector_linewidth=None):
#     """
#     Plot a biplot of PCA scores and loadings.
#     Args:
#         score (np.ndarray): PCA scores.
#         coef (np.ndarray): PCA loadings.
#         eigenvalues (np.ndarray): Eigenvalues.
#         labels (list): List of original labels.
#         colors (list): List of colors for each label.
#         explained_variance (list): List of explained variances.
#         vector_colors (list): List of colors for each vector.
#         scaled (bool): Whether to scale the vectors.
#         vector_linewidth (float): Linewidth for the vectors.

#     Returns:
#         None: Plots the biplot.

#     Raises:
#         TypeError: If the input is not a PCA object.
#     """
#     plt.rcParams.update({'font.size': 20})
#     # Apply 90-degree rotation matrix to `score` and `coef`
#     # rotation_matrix = np.array([[0, -1], [1, 0]])
#     # Rotate the first two components of the scores
#     # score = np.dot(score[:, :2], rotation_matrix)
#     # Rotate the first two components of the loadings
#     # coef = np.dot(coef[:, :2], rotation_matrix)
#     xs = score[:, 0]
#     ys = score[:, 1]
#     n = coef.shape[0]
#     scalex = 1.0 / (xs.max() - xs.min())
#     scaley = 1.0 / (ys.max() - ys.min())

#     padding= 1.2 # 20% padding for axis and vector scaling
#     padding_text = 1.1 # 10 % padding for text
#     xlims = padding * np.max(np.abs(xs))
#     ylims = padding * np.max(np.abs(ys))
#     if colors is None:  # If no color information, plot all points orange
#         plt.scatter(xs, ys, s=80, color='gray', alpha=0.5,edgecolor='gray')
#         # plt.scatter(xs, ys, s=80, color='orange')
#     else:  # If color information is given, plot points with corresponding colors
#         # plt.scatter(xs * scalex, ys * scaley, s=80, color=colors, alpha=0.5)
#         plt.scatter(xs, ys, s=80, color=colors, alpha=0.5,edgecolor=colors)
#     plt.xlim(-xlims * padding,xlims * padding)
#     plt.ylim(-ylims* padding,ylims* padding)

#     # Adjust the number of labels to match the number of coefficients
#     original_labels = labels[:n]

#     # Call the function to get the new mapped labels
#     labels = map_labels(original_labels, custom_mapping)

#     # Draw principal component vectors as arrows
#     for i in range(n):
#         # Calculate arrow end point (arrow head)
#         arrow_end_x = xlims * coef[i, 0]
#         arrow_end_y = ylims * coef[i, 1]
#         if scaled:
#             # Draw the arrow starting from the origin (0, 0) to the arrow head
#             plt.arrow(0, 0, arrow_end_x, arrow_end_y, color=vector_colors[i], alpha=1, linewidth=vector_linewidth,length_includes_head=True,head_length=0.1,head_width=0.1,overhang=0)
#         else:
#             plt.arrow(0, 0, arrow_end_x, arrow_end_y, color=vector_colors[i], alpha=1, linewidth=vector_linewidth,length_includes_head=True,head_length=0.7,head_width=0.7,overhang=0)
#     # Position the label above the arrow head
#         if labels is not None:
#             if arrow_end_y > 0:
#                 va='bottom'
#             else:
#                 va='top'
#             # Annotate the point with text (you can customize the text as needed)
#             plt.text(arrow_end_x * padding_text, arrow_end_y * padding_text,
#                      f"{labels[i]}\n({coef[i, 0]:.2f}, {coef[i, 1]:.2f})",color='black',ha='center', va=va,fontdict=dict(fontsize=14), bbox=dict(facecolor='white', alpha=0.0001, edgecolor='white'))
#     # Write first two eigenvalues
#     # plt.text(-ylims + padding_text, -xlims * padding_text, "Eigenvalues:\n {:.2f}, {:.2f}".format(*eigenvalues),ha='left', va='bottom', fontsize=10)
#     # Write all eigenvalues
#     # plt.text(-ylims + 0.5, -xlims + 0.5, "Eigenvalues:\n " + ", ".join(f"{val:.2f}" for val in eigenvalues), ha='left', va='bottom', fontsize=10)

#     # Set fixed axis limits
#     ax = plt.gca()
#     fixed_x_min, fixed_x_max = -xlims, xlims  # Your specified x limits
#     fixed_y_min, fixed_y_max = -ylims, ylims  # Your specified y limits

#     # Set the fixed limits
#     plt.xlim(fixed_x_min, fixed_x_max)
#     plt.ylim(fixed_y_min, fixed_y_max)

#     # Hide all spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)

#     # Parameters for arrows
#     arrow_length = 0.15  # Size of the arrowhead
#     arrow_linewidth = 2.5

#     # Create y-axis arrow (vertical) ON THE LEFT SPINE
#     #  Start at bottom left corner
#     ax.arrow(fixed_x_min +arrow_length, fixed_y_min+arrow_length,
#              0, fixed_y_max - (fixed_y_max/2.5),  # Go up to max - arrowhead space
#              head_width=arrow_length, head_length=arrow_length,
#              fc='black', ec='black', linewidth=arrow_linewidth,
#              length_includes_head=True)

#     # Create x-axis arrow (horizontal) ON THE BOTTOM SPINE
#     # Start at bottom left corner
#     ax.arrow(fixed_x_min +arrow_length, fixed_y_min+arrow_length,
#              fixed_x_max - (fixed_x_max/2.5), 0,  # Go right to max - arrowhead space
#              head_width=arrow_length, head_length=arrow_length,
#              fc='black', ec='black', linewidth=arrow_linewidth,
#              length_includes_head=True)
#     # Hide tick parameters
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # Remove standard labels
#     ax.set_xlabel('')
#     ax.set_ylabel('')

#     # Add custom positioned labels
#     # X-axis label - at the start of the x-axis arrow
#     ax.text(fixed_x_min + 1.5, fixed_y_min - 0.1,  # Position just below the arrow end
#             "PC{} ({:.1f}%)".format(1, explained_variance[0] * 100),
#             fontsize=18, ha='right', va='top')

#     # Y-axis label - at the start of the y-axis arrow
#     ax.text(fixed_x_min + 0.02, fixed_y_min + 0.6,  # Position to the left of arrow end
#             "PC{} ({:.1f}%)".format(2, explained_variance[1] * 100),
#             fontsize=18, ha='right', va='bottom', rotation=90)

#     # Add a legend for the colors if provided
#     if colors is not None:
#         legend_elements = [
#             Line2D([0], [0],
#                    marker='o',
#                    color='w',
#                    markerfacecolor='#2A586E',
#                    markersize=12,
#                    label='Manual',
#                    alpha=0.8),
#             Line2D([0], [0],
#                    marker='o',
#                    color='w',
#                    markerfacecolor='#cc2936',
#                    markersize=12,
#                    label='Autonomous',
#                    alpha=0.8)
#         ]
#         plt.legend(handles=legend_elements, fontsize=16)

# def get_pca_var(pca, subset_scaled, feature_names):
#     # Validate input types
#     if not isinstance(pca, PCA):
#         raise TypeError("Expected a PCA object from sklearn.decomposition.PCA")

#     # Compute coordinates
#     coords = pca.components_.T * np.sqrt(pca.explained_variance_)

#     # Compute correlations
#     # Since we're using standardized data:
#     cor = coords / np.std(subset_scaled, axis=0)

#     # Compute cos2 (squared loadings or cosine similarity)
#     cos2 = np.square(cor)

#     # Compute contributions
#     total_variance = np.sum(pca.explained_variance_)
#     contrib = (cos2 * 100 * pca.explained_variance_) / total_variance

#     # Create dataframes with feature names as the index
#     coord_df = pd.DataFrame(coords, index=feature_names, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
#     cor_df = pd.DataFrame(cor, index=feature_names, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
#     cos2_df = pd.DataFrame(cos2, index=feature_names, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
#     contrib_df = pd.DataFrame(contrib, index=feature_names, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

#     return {
#         'coord': coord_df,
#         'cor': cor_df,
#         'cos2': cos2_df,
#         'contrib': contrib_df
#     }

# from matplotlib.patches import Circle
# from matplotlib import colors as mcolors

# def plot_cosine_similarity(cos2_df, title='', event='', time_point='', save=False, output_dir: str | None = None, image_format: str = 'pdf'):
#     """
#     Plot the cosine similarity of the PCA components.
#     Args:
#         cos2_df (pd.DataFrame): DataFrame containing cosine similarity values.
#         title (str): Title of the plot.
#         event (str): Event name.
#         time_point (str): Time point.
#         save (bool): Whether to save the plot.

#     Returns:
#         None: Plots the cosine similarity plot.
#     """
#     # Extract feature names directly from the DataFrame index
#     feature_names = cos2_df.index

#     # max_cos2_value = max(cos2_df['PC1'])  # color bar maximum limit
#     max_cos2_value = 1  # color bar maximum limit
#     # Create a custom colormap from white to #A2530E
#     cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "#A2530E"])

#     features, components = cos2_df.shape
#     fig, ax = plt.subplots(figsize=(10, 8))
#     if len(feature_names) > 10:
#         num_fontsize = 10
#     num_fontsize = 12
#     ax.set_title(title, fontsize=18)

#     # Plot each circle and add annotations
#     for i in range(features):
#         for j in range(components):
#             value = cos2_df.iloc[i, j]
#             color = cmap(value / max_cos2_value)  # Normalize the color
#             circle = Circle((j, i), radius=np.sqrt(value) * 0.5, color=color, fill=True)
#             ax.add_artist(circle)

#             # Annotate the cos2 value at the center of the circle
#             ax.text(
#                 j, i, f'{value:.2f}',
#                 color='lightgray',
#                 ha='center', va='center',
#                 fontsize=num_fontsize,  # Adjust font size for readability
#             )

#     # Setup axis limits and labels
#     ax.set_xlim(-0.5, components - 0.5)
#     ax.set_ylim(-0.5, features - 0.5)
#     ax.set_xticks(np.arange(components))
#     ax.set_yticks(np.arange(features))
#     ax.set_xticklabels([f'PC{i+1}' for i in range(components)], rotation=0)
#     ax.set_yticklabels(feature_names)
#     ax.set_aspect('equal', 'box')

#     # Add grid lines for clarity
#     ax.hlines(np.arange(-0.5, features), xmin=-0.5, xmax=components - 0.5, color='grey', lw=0.5)
#     ax.vlines(np.arange(-0.5, components), ymin=-0.5, ymax=features - 0.5, color='grey', lw=0.5)

#     # Set outer border to gray by modifying the spines
#     for spine in ax.spines.values():
#         spine.set_edgecolor('grey')
#         spine.set_linewidth(0.5)

#     # Add a color bar on the right
#     norm = mcolors.Normalize(vmin=0, vmax=max_cos2_value)
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04, label='$Cos^2$')

#     # Increase the number of ticks on the color bar for better granularity
#     num_ticks = 6
#     cbar.set_ticks(np.linspace(0, max_cos2_value, num_ticks))
#     cbar.set_ticklabels([round(val, 2) for val in np.linspace(0, max_cos2_value, num_ticks)])
#     cbar.outline.set_visible(True)
#     plt.tight_layout()
#     if save:
#         if output_dir is None:
#             # Fallback: ensure a safe directory exists under results/cos_plots/<event>
#             _dirs = init_results_dirs(event=event if event else None)
#             output_dir = _dirs['cos_plots']
#         filename = f"pca_cos2_{event or 'all_events'}_{time_point}_{''.join(map(str, feature_names))}.{image_format}"
#         plt.savefig(os.path.join(output_dir, filename), dpi=1200)
#     plt.show()



# # Map each condition to a float number (use in notebook on your df before calling)
# label_mapping = {'Manual': 1, 'Autonomous': 0}
# # with outliers
# resampled_df.loc[:,'conditions'] = resampled_df['Condition'].map(label_mapping)



# Calculate PCA
def visualize_event_pca(df, event='', time_point='', features='', scaled=True, vis_biplot=True, save_biplot=False, save_cos2_plot=False, save_cleaned=False, direc_path='', interactive=True):
    if event != '':
        event_df = df[df['Event'] == event]
        timestamps = event_df['time'].unique()
    else:
        timestamps = df['time'].unique()
    # DataFrames to save data
    df_cleaned_ts = pd.DataFrame()
    df_event_components = pd.DataFrame()
    df_event_variance = pd.DataFrame()
    df_pca_results = pd.DataFrame()
    df_sorted_eigenvalues = pd.DataFrame()  # To store sorted eigenvalues

    # Cache for interactive viewing
    cache = {}
    # Prepare directories if saving any artifacts
    if save_biplot or save_cos2_plot or save_cleaned:
        _dirs = init_results_dirs(event=event if event else None)

    # Updated: Progress bar using tqdm
    with tqdm(total=len(timestamps), desc="Processing times", unit="", colour="green") as pbar:
        start_time = time.time()  # Store the starting time
        for i, timestamp in enumerate(timestamps):
            # 0. Extract rows for this timestamp
            subset = df[df['time'] == timestamp]
            event_name = subset['Event'].unique()[0]

            # 1. Update progress bar for the current timestamp
            pbar.set_postfix_str(f"Current Timestamp: {timestamp}")

            # 2. Clean outliers in subset
            subset_cleaned = subset.pipe(clean_subset)

            # 3. Features
            subset_features1 = subset_cleaned[subset_cleaned.columns.intersection(features)]
            subset_features_renamed = subset_features1.rename(columns=custom_mapping)
            subset_features = subset_features_renamed.drop(columns=['Condition'])
            original_labels = list(subset_features.columns)

            # 3.1 Colors
            colors = [
                '#71898E' if condition == 1 else '#cc2936'  # Manual
                for condition in subset_features_renamed['Condition']
            ]

            # 4. Scaling
            if scaled:
                scaler = StandardScaler(with_std=True)
                subset_scaled = scaler.fit_transform(subset_features)
            else:
                subset_scaled = subset_features

            # 5. Perform PCA
            pca = PCA(whiten=True, svd_solver='full')
            pca_result = pca.fit_transform(subset_scaled)
            explained_variance = pca.explained_variance_ratio_
            eigenvalues = pca.explained_variance_

            # 5.1 Sorted eigenvalues
            sorted_eigenvalue_columns = [f'eigen_val{i+1}' for i in range(len(eigenvalues))]
            eigenvalues_df = pd.DataFrame([eigenvalues], columns=sorted_eigenvalue_columns)
            eigenvalues_df['Time'] = timestamp
            eigenvalues_df['Event'] = event_name
            df_sorted_eigenvalues = pd.concat([df_sorted_eigenvalues, eigenvalues_df], ignore_index=True)

            # Save PCA results
            pca_result_df = pd.DataFrame(
                pca_result,
                columns=[f'PC{i+1}' for i in range(len(explained_variance))]
            )
            pca_result_df['Time'] = timestamp
            pca_result_df['Event'] = event_name
            df_pca_results = pd.concat([df_pca_results, pca_result_df], ignore_index=True)

            # Components
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i, v in enumerate(explained_variance)],
            )
            loadings['Features'] = original_labels
            loadings['Time'] = timestamp
            loadings['Event'] = event_name
            df_event_components = pd.concat([df_event_components, loadings], ignore_index=True)

            # Variance
            df_variances = pd.DataFrame(
                np.reshape(explained_variance, (1, len(explained_variance))),
                columns=[f'PC{i+1}' for i, v in enumerate(explained_variance)]
            )
            df_variances['Time'] = timestamp
            df_variances['Event'] = event_name
            df_event_variance = pd.concat([df_event_variance, df_variances], ignore_index=True)

            # Save plots immediately if requested (slider is for viewing only)
            if save_biplot:
                biplot_dir = _dirs['biplots']
                sns.set_theme(style="white")
                plt.figure(figsize=(14, 10))
                plt.title('Second: ' + str(timestamp), fontsize=22)
                biplot(
                    pca_result,
                    pca.components_.T,
                    eigenvalues,
                    original_labels,
                    colors,
                    explained_variance,
                    vector_colors=['k'] * len(original_labels),
                    scaled=scaled,
                    vector_linewidth=1.1
                )
                filename = f"pca_{event or 'all_events'}_{time_point}_{''.join(map(str, original_labels))}.pdf"
                plt.savefig(os.path.join(biplot_dir, filename), dpi=1200)
                plt.close()

            if save_cos2_plot:
                pca_cos2 = get_pca_var(pca, subset_scaled, original_labels)
                cos2_matrix = pca_cos2['cos2']
                plot_cosine_similarity(cos2_matrix, title='Cosine Similarity ($Cos^2$)', save=True, event=event, time_point=time_point, output_dir=_dirs['cos_plots'])
                plt.close()

            # Visualization for non-interactive mode only
            if vis_biplot and not interactive:
                sns.set_theme(style="white")
                plt.figure(figsize=(14, 10))
                plt.title('Second: ' + str(timestamp), fontsize=22)
                biplot(
                    pca_result,
                    pca.components_.T,
                    eigenvalues,
                    original_labels,
                    colors,
                    explained_variance,
                    vector_colors=['k'] * len(original_labels),
                    scaled=scaled,
                    vector_linewidth=1.1
                )
                plt.show()

                pca_cos2 = get_pca_var(pca, subset_scaled, original_labels)
                cos2_matrix = pca_cos2['cos2']
                plot_cosine_similarity(cos2_matrix, title='Cosine Similarity ($Cos^2$)', save=False, event=event, time_point=time_point)

            # Cleaned subset
            if save_cleaned:
                subset_features_renamed['Event'] = event_name
                subset_features_renamed['Time'] = timestamp
                df_cleaned_ts = pd.concat([df_cleaned_ts, subset_features_renamed], ignore_index=True)

            # Store in cache for interactive viewing
            cache[i] = {
                'timestamp': timestamp,
                'original_labels': original_labels,
                'colors': colors,
                'pca': pca,
                'pca_result': pca_result,
                'explained_variance': explained_variance,
                'eigenvalues': eigenvalues,
                'subset_scaled': subset_scaled,
            }

            # Update progress bar and elapsed time
            elapsed_time = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix_str(f"Elapsed: {elapsed_time:.2f}s")

    # Save results
    if save_cleaned:
        if len(timestamps) > 1:
            timestamp = 'all'
        # Ensure event-specific data directories exist
        data_dirs = init_results_dirs(event=event if event else None)
        df_cleaned_ts.to_csv(os.path.join(data_dirs['cleaned_new'], f"cleaned_event_{event or 'all_events'}_timestamp_{str(timestamp)}_{len(original_labels)}_variables.csv"), index=False)
        df_event_variance.to_csv(os.path.join(data_dirs['variances'], f"variances_event_{event or 'all_events'}_timestamp_{str(timestamp)}_{len(original_labels)}_variables.csv"), index=False)
        df_event_components.to_csv(os.path.join(data_dirs['components'], f"components_event_{event or 'all_events'}_timestamp_{str(timestamp)}_{len(original_labels)}_variables.csv"), index=False)
        df_sorted_eigenvalues.to_csv(os.path.join(data_dirs['eigenvalues'], f"eigenval_event_{event or 'all_events'}_timestamp_{str(timestamp)}_{len(original_labels)}_variables.csv"), index=False)

    # Interactive slider viewer (plots stacked) - viewing only, no saving here
    if interactive and vis_biplot and len(timestamps) > 0:
        import ipywidgets as widgets
        from IPython.display import display

        slider = widgets.IntSlider(
            min=0,
            max=len(timestamps)-1,
            step=1,
            value=0,
            description='Index',
            layout=widgets.Layout(width='800px'),
            continuous_update=True,
        )

        def render(idx: int):
            comp = cache.get(idx)
            if comp is None:
                return
            sns.set_theme(style="white")
            plt.figure(figsize=(14, 10))
            plt.title('Second: ' + str(comp['timestamp']), fontsize=22)
            biplot(
                comp['pca_result'],
                comp['pca'].components_.T,
                comp['eigenvalues'],
                comp['original_labels'],
                comp['colors'],
                comp['explained_variance'],
                vector_colors=['k'] * len(comp['original_labels']),
                scaled=scaled,
                vector_linewidth=1.1
            )
            plt.show()

            pca_cos2 = get_pca_var(comp['pca'], comp['subset_scaled'], comp['original_labels'])
            cos2_matrix = pca_cos2['cos2']
            plot_cosine_similarity(cos2_matrix, title='Cosine Similarity ($Cos^2$)', save=False, event=event, time_point=time_point)

        out = widgets.interactive_output(render, {'idx': slider})
        display(slider, out)

    return df_event_components, df_event_variance, original_labels, df_cleaned_ts, df_pca_results, df_sorted_eigenvalues

    # Note: In a notebook, you can build an interactive viewer using the cached results.



event = 'one'
scaled = True
features = conditions + euler_features

components_df, variances_df, original_labels, df_cleaned_ts,df_pca_results, eigenvalues_df = visualize_event_pca(resampled_df, event=event, 
features=features, scaled=scaled, vis_biplot=True, save_biplot=False, save_cos2_plot=False, save_cleaned=False)#, direc_path=directories)


# import ipywidgets as widgets  # Add import if not already present
# import logging
# logging.getLogger('matplotlib.font_manager').disabled = True

# import ipywidgets as widgets  # interactive display
# # %config InlineBackend.figure_format = 'svg'
# plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

# event = 'one'
# time_point = 'dynamic'
# scaled = True
# timestamps = sorted(resampled_df[resampled_df['Event'] == event]['time'].unique())
# @widgets.interact(d_ts=widgets.FloatSlider(value=0.2, min=-5, max=5, step=0.02,description='Time',layout=widgets.Layout(width='800px')))

# def interactive_pca(d_ts):
#     components_df, variances_df, original_labels, df_cleaned_ts, df_pca_results, eigenvalues_df = visualize_event_pca(
#         resampled_df,
#         event=event,
#         time_point=time_point,
#         d_ts=d_ts, # dynamic timestamp for slider
#         features=features,
#         scaled=scaled,
#         vis_biplot=True,
#         output_mode='show',
#         legend_mode='condition',
#         save_biplot=True,
#         save_cos2_plot=True,
#         save_cleaned=True,
#         direc_path=directories
#     )
