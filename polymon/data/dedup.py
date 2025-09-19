from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from rdkit import Chem
from sklearn.linear_model import LinearRegression

from polymon.setting import REPO_DIR


class Dedup:
    def __init__(
        self, 
        df: pd.DataFrame,
        label: str, 
        must_keep: List[str] = None,
        rtol: float = 0.05,
    ):
        """Deduplicate the dataframe. The input dataframe should have the 
        following columns: SMILES, Source, label, id.

        Args:
            df (pd.DataFrame): The dataframe to deduplicate.
            label (str): The label column name.
            must_keep (List[str]): The sources to keep.
            rtol (float): The relative tolerance for deduplication.
        """
        self.label = label
        self.rtol = rtol
        # Remove other columns
        df = df[['SMILES', 'Source', label, 'id']]
        self.df = df.dropna(subset=[label])
        self.must_keep = must_keep or []
    
    def add_df(
        self,
        file_path: str,
        smiles_col: str,
        label_col: str,
        source: str,
    ):
        """Add a dataframe to the :obj:`self.df`.

        Args:
            file_path (str): The path to the dataframe.
            smiles_col (str): The column name of the SMILES.
            label_col (str): The column name of the label.
            source (str): The source name.
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f'Unsupported file type: {file_path}')
        assert label_col in df.columns and smiles_col in df.columns,\
            f'Check the columns {df.columns.tolist()}'
        df = df[df[label_col].notna()]
        df = df[[smiles_col, label_col]].assign(
            Source=source,
        )
        df.rename(columns={
            label_col: self.label, 
            smiles_col: 'SMILES'
        }, inplace=True)
        df['SMILES'] = df['SMILES'].apply(lambda x: Chem.CanonSmiles(x))
        self.df = pd.concat([self.df, df], ignore_index=True)
    
    def show_distribution(
        self, 
        sources: List[str],
        bins: int = 50, 
        alpha: float = 0.5,
        density: bool = False,
    ):
        """Plot the distribution of the label for each source.

        Args:
            sources (List[str]): The sources to plot.
            bins (int): The number of bins.
            alpha (float): The alpha of the histogram.
            density (bool): Whether to plot the density.
        """
        # Plot the distribution of the label for each source
        for source in sources:
            df = self.df[self.df['Source'] == source]
            plt.hist(
                df[self.label], 
                bins=bins, 
                alpha=alpha, 
                label=source, 
                density=density,
            )
        plt.legend()
        plt.xlabel(self.label)
        plt.ylabel('Density' if density else 'Count')
        plt.show()
    
    def run(self, sources: Optional[List[str]] = None, save: bool = False):
        """Run the deduplication. The deduplication is performed by the 
        following steps:

        1. Remove rows with higher relative difference.
        2. Keep indices if the source is in :obj:`self.must_keep`.

        Args:
            sources (Optional[List[str]]): The sources to deduplicate.
            save (bool): Whether to save the deduplicated dataframe.
        """
        if sources is None:
            sources = self.df['Source'].unique()
        sources = list(set(sources + self.must_keep))
        source_info = [
            f'{source}: {len(self.df[self.df["Source"] == source])}' \
            for source in sources
        ]
        logger.info(f'Sources: {", ".join(source_info)}')
        df = self.df[self.df['Source'].isin(sources)]
        
        logger.info(f'Number of rows before deduplication: {len(df)}')
        mask = df['SMILES'].duplicated(keep=False)
        duplicates = df[mask]
        smiles_groups = duplicates.groupby('SMILES').groups
        drop_indices = []
        for smiles, indices in smiles_groups.items():
            rows = df.loc[indices]
            
            # 1. Remove rows with higher relative difference.
            if len(rows) > 1:
                v = rows[self.label]
                if np.abs(v.max() - v.min()) / (v.max() + 1e-6) > self.rtol:
                    drop_indices.extend(rows.index)
                else:
                    df.loc[rows.index[0], self.label] = v.mean()
                    drop_indices.extend(rows.index[1:])
        
        # 2. Keep indices if the source is in must_keep
        if self.must_keep is not None:
            for source in self.must_keep:
                must_keep_indices = df[df['Source'] == source].index
                drop_indices = list(set(drop_indices) - set(must_keep_indices))
        
        df = df.drop(drop_indices)
        logger.info(f'Number of rows after deduplication: {len(df)}')
        
        if save:
            path = str(REPO_DIR / 'database' / 'merged' / f'{self.label}_{"_".join(sources)}.csv')
            df['id'] = np.arange(len(df))
            df.to_csv(path, index=False)
        
        return df
    
    def compare(
        self,
        source1: str,
        source2: str,
        fitting: bool = False,
    ):
        """Compare the label of the two sources with the same smiles.

        Args:
            source1 (str): The first source name.
            source2 (str): The second source name.
            fitting (bool): Whether to fit the data. If True, the function will
                fit the data with a linear regression and add the fitted points 
                as a new source.
        """
        # Compare the label of the two sources with the same smiles
        df1 = self.df[self.df['Source'] == source1]
        df2 = self.df[self.df['Source'] == source2]
        df = pd.merge(df1, df2, on='SMILES', how='inner')
        x = df[self.label + '_x'].values
        y = df[self.label + '_y'].values
        
        # Plot the comparison
        nan_mask = np.isnan(x) | np.isnan(y)
        x = x[~nan_mask]
        y = y[~nan_mask]
        plt.scatter(x, y, color='blue', alpha=0.5, label='Data')
        plt.xlabel(f'{self.label} ({source1})')
        plt.ylabel(f'{self.label} ({source2})')
        plt.gca().set_aspect('equal')
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        if fitting:
            # Learn linear relationship y = ax + b to fit the data
            model = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
            y_pred = model.predict(x.reshape(-1, 1))
            logger.info(f'Coefficient: {model.coef_[0][0]:.4f}, Intercept: {model.intercept_[0]:.4f}')
            plt.scatter(y_pred, y, color='red', alpha=0.5, marker='x', label='Fitted')
            # Write the equation of the fitted line
            plt.text(
                min_val, max_val, 
                f'y = {model.coef_[0][0]:.4f}x + {model.intercept_[0]:.4f}',
                fontsize=12,
                ha='left',
            )
            plt.legend()
        
        plt.show()
        if fitting:
            # Add the fitted points as a new source
            df = self.df[self.df['Source'] == source2]
            y_fitted = model.predict(df[self.label].values.reshape(-1, 1))
            df_fitted = pd.DataFrame({
                'SMILES': df['SMILES'],
                self.label: y_fitted.squeeze(-1),
                'Source': f'{source2}_fitted',
            })
            self.df = pd.concat([self.df, df_fitted], ignore_index=True)
        return None