import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DataAnalyzer:
    def __init__(self, dataset_path=None):
        """
        Initialize the DataAnalyzer with a dataset path.
        If no path is provided, will use the Iris dataset as fallback.
        """
        self.dataset_path = dataset_path
        self.data = None
        self.cleaned_data = None

    def load_dataset(self):
        """
        Load dataset from CSV file with error handling.
        Falls back to Iris dataset if CSV file is not found.
        """
        try:
            if self.dataset_path and self.dataset_path.endswith('.csv'):
                print(f"Loading dataset from: {self.dataset_path}")
                self.data = pd.read_csv(self.dataset_path)
                print("✅ Dataset loaded successfully from CSV file!")
            else:
                raise FileNotFoundError("No valid CSV file provided")

        except FileNotFoundError:
            print("⚠️  CSV file not found. Using Iris dataset as fallback...")
            iris = load_iris()
            self.data = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.data['species'] = iris.target
            self.data['species'] = self.data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            print("✅ Iris dataset loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False

        return True

    def inspect_data(self):
        """
        Display basic information about the dataset.
        """
        if self.data is None:
            print("❌ No data loaded. Please load a dataset first.")
            return

        print("\n" + "=" * 50)
        print("📊 DATASET INSPECTION")
        print("=" * 50)

        # Display first few rows
        print("\n🔍 First 5 rows of the dataset:")
        print(self.data.head())

        # Display dataset info
        print(f"\n📏 Dataset shape: {self.data.shape}")
        print(f"📋 Columns: {list(self.data.columns)}")

        # Data types
        print("\n📊 Data types:")
        print(self.data.dtypes)

        # Missing values
        print("\n❓ Missing values:")
        missing_values = self.data.isnull().sum()
        if missing_values.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing_values[missing_values > 0])

        # Basic statistics
        print("\n📈 Basic statistics:")
        print(self.data.describe())

    def clean_data(self):
        """
        Clean the dataset by handling missing values.
        """
        if self.data is None:
            print("❌ No data loaded. Please load a dataset first.")
            return

        print("\n" + "=" * 50)
        print("🧹 DATA CLEANING")
        print("=" * 50)

        self.cleaned_data = self.data.copy()

        # Check for missing values
        missing_before = self.cleaned_data.isnull().sum().sum()

        if missing_before > 0:
            print(f"Found {missing_before} missing values. Cleaning data...")

            # For numerical columns, fill with median
            numerical_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.cleaned_data[col].isnull().sum() > 0:
                    median_val = self.cleaned_data[col].median()
                    self.cleaned_data[col].fillna(median_val, inplace=True)
                    print(f"  • Filled missing values in '{col}' with median: {median_val:.2f}")

            # For categorical columns, fill with mode
            categorical_cols = self.cleaned_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.cleaned_data[col].isnull().sum() > 0:
                    mode_val = self.cleaned_data[col].mode()[0] if not self.cleaned_data[
                        col].mode().empty else 'Unknown'
                    self.cleaned_data[col].fillna(mode_val, inplace=True)
                    print(f"  • Filled missing values in '{col}' with mode: {mode_val}")

            missing_after = self.cleaned_data.isnull().sum().sum()
            print(f"✅ Data cleaning complete! Missing values: {missing_before} → {missing_after}")
        else:
            print("✅ No missing values found. Data is already clean!")

    def basic_analysis(self):
        """
        Perform basic statistical analysis and grouping operations.
        """
        if self.cleaned_data is None:
            print("❌ No cleaned data available. Please clean the data first.")
            return

        print("\n" + "=" * 50)
        print("📊 BASIC DATA ANALYSIS")
        print("=" * 50)

        # Basic statistics
        print("\n📈 Detailed statistics for numerical columns:")
        numerical_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        print(self.cleaned_data[numerical_cols].describe())

        # Grouping analysis
        categorical_cols = self.cleaned_data.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            print(f"\n🔍 Grouping analysis:")
            cat_col = categorical_cols[0]  # Use first categorical column
            num_col = numerical_cols[0]  # Use first numerical column

            print(f"Grouping by '{cat_col}' and computing mean of '{num_col}':")
            grouped_data = self.cleaned_data.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).round(2)
            print(grouped_data)

            # Find interesting patterns
            print(f"\n🔍 Interesting findings:")
            if len(grouped_data) > 1:
                max_group = grouped_data['mean'].idxmax()
                min_group = grouped_data['mean'].idxmin()
                print(f"  • Highest average {num_col}: {max_group} ({grouped_data.loc[max_group, 'mean']:.2f})")
                print(f"  • Lowest average {num_col}: {min_group} ({grouped_data.loc[min_group, 'mean']:.2f})")

                # Calculate coefficient of variation
                cv = grouped_data['std'] / grouped_data['mean']
                most_variable = cv.idxmax()
                print(f"  • Most variable group: {most_variable} (CV: {cv[most_variable]:.2f})")
        else:
            print("⚠️  No suitable categorical and numerical columns found for grouping analysis.")

    def create_visualizations(self):
        """
        Create four different types of visualizations.
        """
        if self.cleaned_data is None:
            print("❌ No cleaned data available. Please clean the data first.")
            return

        print("\n" + "=" * 50)
        print("📊 DATA VISUALIZATION")
        print("=" * 50)

        numerical_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.cleaned_data.select_dtypes(include=['object']).columns

        if len(numerical_cols) == 0:
            print("❌ No numerical columns found for visualization.")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Line Chart (Time series or trend)
        plt.subplot(2, 2, 1)
        if len(numerical_cols) >= 2:
            # Use first two numerical columns for line plot
            x_col = numerical_cols[0]
            y_col = numerical_cols[1]
            plt.plot(self.cleaned_data[x_col], self.cleaned_data[y_col],
                     linewidth=2, marker='o', markersize=4, alpha=0.7)
            plt.title(f'Line Chart: {y_col} vs {x_col}', fontsize=14, fontweight='bold')
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.grid(True, alpha=0.3)
        else:
            # Single numerical column as time series
            plt.plot(self.cleaned_data[numerical_cols[0]], linewidth=2, alpha=0.7)
            plt.title(f'Line Chart: {numerical_cols[0]} Trend', fontsize=14, fontweight='bold')
            plt.xlabel('Index', fontsize=12)
            plt.ylabel(numerical_cols[0], fontsize=12)
            plt.grid(True, alpha=0.3)

        # 2. Bar Chart (Categorical comparison)
        plt.subplot(2, 2, 2)
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numerical_cols[0]
            grouped = self.cleaned_data.groupby(cat_col)[num_col].mean().sort_values(ascending=False)

            bars = plt.bar(range(len(grouped)), grouped.values,
                           color=plt.cm.Set3(np.linspace(0, 1, len(grouped))))
            plt.title(f'Bar Chart: Average {num_col} by {cat_col}', fontsize=14, fontweight='bold')
            plt.xlabel(cat_col, fontsize=12)
            plt.ylabel(f'Average {num_col}', fontsize=12)
            plt.xticks(range(len(grouped)), grouped.index, rotation=45, ha='right')

            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        else:
            # Fallback: histogram of first numerical column
            plt.hist(self.cleaned_data[numerical_cols[0]], bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'Bar Chart: Distribution of {numerical_cols[0]}', fontsize=14, fontweight='bold')
            plt.xlabel(numerical_cols[0], fontsize=12)
            plt.ylabel('Frequency', fontsize=12)

        # 3. Histogram (Distribution)
        plt.subplot(2, 2, 3)
        if len(numerical_cols) >= 1:
            plt.hist(self.cleaned_data[numerical_cols[0]], bins=20, alpha=0.7,
                     color='skyblue', edgecolor='black', linewidth=0.5)
            plt.title(f'Histogram: Distribution of {numerical_cols[0]}', fontsize=14, fontweight='bold')
            plt.xlabel(numerical_cols[0], fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Add mean line
            mean_val = self.cleaned_data[numerical_cols[0]].mean()
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_val:.2f}')
            plt.legend()

        # 4. Scatter Plot (Relationship between two numerical variables)
        plt.subplot(2, 2, 4)
        if len(numerical_cols) >= 2:
            x_col = numerical_cols[0]
            y_col = numerical_cols[1]

            # Create scatter plot with different colors for different groups if categorical column exists
            if len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                unique_categories = self.cleaned_data[cat_col].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))

                for i, category in enumerate(unique_categories):
                    mask = self.cleaned_data[cat_col] == category
                    plt.scatter(self.cleaned_data[mask][x_col],
                                self.cleaned_data[mask][y_col],
                                c=[colors[i]], label=category, alpha=0.7, s=50)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.scatter(self.cleaned_data[x_col], self.cleaned_data[y_col],
                            alpha=0.7, s=50, color='steelblue')

            plt.title(f'Scatter Plot: {y_col} vs {x_col}', fontsize=14, fontweight='bold')
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.grid(True, alpha=0.3)

            # Add correlation coefficient
            correlation = self.cleaned_data[x_col].corr(self.cleaned_data[y_col])
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                     transform=plt.gca().transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Fallback: box plot
            plt.boxplot([self.cleaned_data[numerical_cols[0]]], labels=[numerical_cols[0]])
            plt.title(f'Box Plot: Distribution of {numerical_cols[0]}', fontsize=14, fontweight='bold')
            plt.ylabel(numerical_cols[0], fontsize=12)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("✅ All visualizations created successfully!")

    def run_complete_analysis(self):
        """
        Run the complete data analysis pipeline.
        """
        print("🚀 Starting Complete Data Analysis Pipeline")
        print("=" * 60)

        # Step 1: Load dataset
        if not self.load_dataset():
            return

        # Step 2: Inspect data
        self.inspect_data()

        # Step 3: Clean data
        self.clean_data()

        # Step 4: Basic analysis
        self.basic_analysis()

        # Step 5: Create visualizations
        self.create_visualizations()

        print("\n" + "=" * 60)
        print("✅ Complete data analysis pipeline finished successfully!")
        print("=" * 60)


def main():
    """
    Main function to run the data analysis.
    """
    print("📊 Welcome to the Data Analysis Tool!")
    print("=" * 50)

    # Try to load from CSV file first
    csv_path = "dataset.csv"  # Change this to your CSV file path

    # Initialize analyzer
    analyzer = DataAnalyzer(csv_path)

    # Run complete analysis
    analyzer.run_complete_analysis()

    # Additional option to save cleaned data
    if analyzer.cleaned_data is not None:
        save_option = input("\n💾 Would you like to save the cleaned dataset? (y/n): ").lower()
        if save_option == 'y':
            output_path = "cleaned_dataset.csv"
            analyzer.cleaned_data.to_csv(output_path, index=False)
            print(f"✅ Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
  main()