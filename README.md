# InferEcon: Python-Based Econometrics Toolkit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://inferEcon.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/insdaguirre/InferEcon)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Python toolkit that replicates Stata's core econometric functionality through an intuitive Streamlit GUI interface. Built for researchers, students, and practitioners who want the power of Stata with the flexibility of Python.

## 🎯 **Project Rationale & Design Philosophy**

### **Why This Toolkit Exists**

Traditional econometric software like Stata, while powerful, has several limitations:
- **Cost**: Expensive licenses for individuals and institutions
- **Platform Lock-in**: Windows-only, limited cloud deployment
- **Closed Source**: Cannot extend or modify core functionality
- **Reproducibility**: Limited integration with modern data science workflows

### **Design Decisions & Justifications**

#### **1. Plugin Architecture (`analysis_functions/`)**
- **Modularity**: Each Stata command is a separate Python module, making maintenance and extension simple
- **Discoverability**: Functions are automatically loaded based on naming convention (`*_func.py`)
- **Scalability**: New functions can be added without modifying core application code
- **Testing**: Individual functions can be tested in isolation

#### **2. Streamlit GUI Interface**
- **Accessibility**: Web-based interface works on any device with a browser
- **Modern UX**: Clean, responsive design that's familiar to Stata users
- **Real-time Preview**: See results immediately without waiting for batch processing
- **Export Capabilities**: Generate HTML reports for sharing and documentation

#### **3. Python Ecosystem Integration**
- **Rich Libraries**: Leverages pandas, numpy, scipy, statsmodels, and scikit-learn
- **Reproducibility**: Full Python code execution with version-controlled dependencies
- **Extensibility**: Easy to add custom statistical methods or integrate with other tools
- **Cloud Ready**: Can be deployed on any platform that supports Python

#### **4. Stata Command Parity**
- **Familiar Syntax**: Functions replicate Stata's command structure and output format
- **Comprehensive Coverage**: Includes descriptive statistics, regression diagnostics, panel data, and advanced econometrics
- **Interpretation**: Provides not just results, but explanations and guidance

## 🚀 **Features**

### **Descriptive Statistics**
- `summarize` - Basic summary statistics (N, mean, std dev, min, max)
- `mean` - Means with standard errors and confidence intervals
- `tabstat` - Custom summary statistics in configurable tables
- `detail` - Comprehensive statistics including percentiles, skewness, kurtosis

### **Data Exploration**
- `describe` - Variable properties and dataset overview
- `list` - Data preview with intelligent sampling
- `tabulate` - Frequency tables and cross-tabulations
- `correlate` - Correlation matrices with significance indicators

### **Regression & Diagnostics**
- `regress` - OLS regression with comprehensive diagnostics
- `estat vif` - Multicollinearity detection
- `estat hettest` - Heteroskedasticity tests
- `estat ovtest` - Ramsey RESET functional form tests
- `estat ic` - Model comparison (AIC/BIC)
- `estat gof` - Goodness of fit statistics
- `linktest` - Model specification error detection

### **Visualization**
- `scatter` - Scatter plots with trend analysis
- `twoway` - Regression lines with confidence intervals
- `avplot` - Added-variable plots (partial regression)
- `margins` - Marginal effects and interaction plots

### **Advanced Econometrics**
- `ivregress` - Instrumental variables (2SLS)
- `hausman` - Fixed vs Random effects comparison
- `xtreg` - Panel data regressions (FE/RE/BE/FD)
- `areg` - High-dimensional fixed effects absorption

### **Automation & Workflows**
- `codebook` - Comprehensive variable documentation
- `bysort` - Grouped analysis and comparisons
- `collapse` - Data aggregation by groups

## ⚙️ **Interactive Configuration Interface**

### **Smart Function Configuration**
Many functions now include **interactive configuration menus** that appear when you select them, allowing you to:

- **Specify Variables**: Choose exactly which variables to use for each analysis
- **Set Parameters**: Configure grouping methods, aggregation functions, and other options
- **Customize Analysis**: Tailor each function to your specific research needs

### **Functions with Configuration Menus**

#### **Regression & Analysis Functions**
- **`regress`** - Select dependent and independent variables
- **`avplot`** - Choose variable to plot and control variables
- **`margins`** - Specify variables for marginal effects analysis

#### **Panel Data Functions**
- **`xtreg`** - Configure panel structure and variable selection
- **`areg`** - Set up fixed effects and variable specifications
- **`hausman`** - Define panel structure for FE vs RE comparison

#### **Instrumental Variables**
- **`ivregress`** - Select dependent variable, endogenous variable, instrument, and controls

#### **Grouped Analysis**
- **`bysort`** - Choose grouping variable, analysis variable, and grouping method
- **`collapse`** - Select grouping variables, aggregation variables, and functions

### **Configuration Features**
- **Smart Detection**: Only shows configuration for functions that need it
- **User-Friendly Interface**: Clear dropdowns, multi-select boxes, and sliders
- **Data Type Awareness**: Automatically detects numeric vs categorical variables
- **Validation**: Ensures your selections are valid before running analysis
- **Flexible Options**: Multiple grouping methods, aggregation functions, and parameter settings

## 🛠️ **Technical Architecture**

### **Core Components**
```
InferEcon/
├── analysis_functions/          # Plugin system for analysis functions
│   ├── __init__.py             # Dynamic function discovery
│   ├── *_func.py               # Individual Stata command implementations
│   └── ...
├── app/
│   └── streamlit_app.py        # Main GUI application
├── run_app.py                  # Environment setup and launcher
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### **Plugin System Design**
The plugin system uses Python's dynamic import capabilities to automatically discover and load analysis functions:

```python
# Functions are discovered based on naming convention
# Each function module must provide:
display_name: str          # Human-readable name for GUI
apply(df: pd.DataFrame)    # Main function that processes data
```

### **Data Flow**
1. **Upload**: CSV files are loaded into pandas DataFrames
2. **Selection**: Users choose analysis functions from dynamic list
3. **Processing**: Selected functions are executed with data
4. **Output**: Results are formatted as tables, plots, or text
5. **Export**: Complete analysis can be exported as HTML report

## 📊 **Usage Examples**

### **Interactive Workflow**
1. **Upload Data**: Load your CSV file through the web interface
2. **Select Functions**: Choose which analyses to run from the sidebar
3. **Configure Parameters**: **NEW!** Specify variables and settings for each function
4. **Run Analysis**: Execute all selected functions with your configurations
5. **View Results**: See tables, plots, and interpretations
6. **Export Report**: Download complete HTML report

### **Configuration Example**
When you select functions like `regress` or `bysort`, configuration menus appear:

**For Regression Analysis:**
- Select dependent variable (Y) from dropdown
- Choose independent variables (X) with multi-select
- Automatic validation ensures valid selections

**For Grouped Analysis:**
- Pick grouping variable (e.g., income quartiles)
- Select variable to analyze (e.g., test scores)
- Choose grouping method (quartiles, quintiles, custom bins)

### **Programmatic Usage**
```python
# The GUI handles this automatically, but here's what happens:
from analysis_functions.summarize_func import apply as summarize
from analysis_functions.regress_func import apply as regress

# Load data
df = pd.read_csv('your_data.csv')

# Run analyses with configuration
config = {'y_col': 'income', 'x_cols': ['education', 'experience']}
regression_results = regress(df, config)

# Results are automatically formatted and displayed
```

### **Adding New Functions**
To add a new Stata command equivalent:

1. Create `analysis_functions/newcommand_func.py`
2. Implement the required interface:
   ```python
   display_name = "New Command"
   
   def apply(df: pd.DataFrame, config: dict = None) -> List[dict]:
       # Your analysis logic here
       # Use config parameter for user-specified variables/parameters
       return [{"type": "table", "title": "Results", "data": results_df}]
   
   def apply_with_config(df: pd.DataFrame, config: dict) -> List[dict]:
       """Optional: For functions that need configuration"""
       return apply(df, config)
   ```
3. **For functions needing configuration**, add to `config_interface.py`:
   ```python
   FUNCTION_CONFIGS = {
       'New Command': YourConfigClass,
       # ... existing configs
   }
   ```
4. The function automatically appears in the GUI with configuration support

## 🌐 **Live Demo**

**Try InferEcon online:** [Live Demo](https://inferecon.streamlit.app/)

*Your econometrics toolkit is now live and ready to use!*

## 🔧 **Installation & Setup**

### **Requirements**
- Python 3.8+
- See `requirements.txt` for specific package versions

### **Quick Start**
```bash
# Clone repository
git clone https://github.com/insdaguirre/InferEcon.git
cd InferEcon

# Install dependencies
pip install -r requirements.txt

# Launch application
python run_app.py
```

### **Alternative Launch**
```bash
# Direct Streamlit launch (if environment is properly configured)
streamlit run app/streamlit_app.py
```

## 🎨 **Design Principles**

### **1. Stata Compatibility**
- **Command Parity**: Each function replicates Stata's behavior and output
- **Familiar Output**: Tables and plots match Stata's presentation style
- **Interpretation**: Results include guidance similar to Stata's help system

### **2. User Experience**
- **Intuitive Interface**: Clear function selection and result preview
- **Immediate Feedback**: Real-time results without waiting
- **Export Options**: Multiple output formats for different use cases

### **3. Extensibility**
- **Plugin Architecture**: Easy to add new functions without core changes
- **Standardized Interface**: Consistent API for all analysis functions
- **Documentation**: Clear examples and usage patterns

### **4. Performance**
- **Efficient Processing**: Leverages optimized Python libraries
- **Memory Management**: Handles large datasets gracefully
- **Caching**: Results are cached for quick access

## 🔬 **Statistical Rigor**

### **Methodology**
- **Peer-Reviewed Libraries**: Uses established statistical packages (scipy, statsmodels)
- **Best Practices**: Implements standard econometric procedures
- **Validation**: Results are verified against known datasets and Stata outputs

### **Quality Assurance**
- **Error Handling**: Graceful degradation when data is unsuitable
- **Input Validation**: Checks for appropriate data types and sample sizes
- **Interpretation**: Provides guidance on result interpretation and limitations

## 🚀 **Future Development**

### **Planned Features**
- **Time Series Analysis**: ARIMA, VAR, and other time series methods
- **Non-Parametric Methods**: Kernel regression, density estimation
- **Machine Learning Integration**: LASSO, Ridge, Elastic Net
- **Database Connectivity**: Direct connection to SQL databases
- **Collaborative Features**: Share analyses and results

### **Extension Points**
- **Custom Functions**: User-defined analysis methods
- **API Interface**: Programmatic access to toolkit functions
- **Plugin Marketplace**: Community-contributed functions
- **Cloud Deployment**: AWS, Google Cloud, Azure integration

## 🤝 **Contributing**

This toolkit is designed for community contribution. To add new functions:

1. **Fork** the repository
2. **Create** your function following the plugin pattern
3. **Test** with various datasets
4. **Submit** a pull request with documentation

### **Development Guidelines**
- Follow the existing naming conventions
- Include comprehensive error handling
- Provide clear output formatting
- Add appropriate documentation

## 📚 **References & Acknowledgments**

### **Statistical Methods**
- **OLS Regression**: Based on Greene's Econometric Analysis
- **Panel Data**: Following Wooldridge's Econometric Analysis of Cross Section and Panel Data
- **Instrumental Variables**: Based on Angrist & Pischke's Mostly Harmless Econometrics

### **Python Libraries**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Statistical functions
- **statsmodels**: Econometric modeling
- **scikit-learn**: Machine learning utilities
- **matplotlib/seaborn**: Data visualization
- **streamlit**: Web application framework

## 📄 **License**

This project is open source and available under the MIT License.

## 🆘 **Support & Community**

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions about econometric methods
- **Documentation**: Comprehensive examples and tutorials
- **Contributions**: Help improve the toolkit for everyone

---

**InferEcon**: Bringing the power of Stata to the Python ecosystem with modern, extensible design.
